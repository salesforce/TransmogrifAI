/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.AllowLabelAsInput
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichVector._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.Metadata

import scala.reflect.runtime.universe.TypeTag

/**
 * Smart bucketizer for numeric map values based on a Decision Tree classifier.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti2          type tag for numeric feature type
 * @param ttiv2         type tag for numeric feature value type
 * @param nev           numeric evidence for feature type value
 * @tparam N  numeric feature type value
 * @tparam I2 numeric map feature type
 */
class DecisionTreeNumericMapBucketizer[N, I2 <: OPMap[N]]
(
  operationName: String = "dtNumMapBuck",
  uid: String = UID[DecisionTreeNumericMapBucketizer[_, _]]
)(
  implicit tti2: TypeTag[I2],
  ttiv2: TypeTag[I2#Value],
  val nev: Numeric[N]
) extends BinaryEstimator[RealNN, I2, OPVector](operationName = operationName, uid = uid)
  with DecisionTreeNumericBucketizerParams
  with VectorizerDefaults with TrackInvalidParam
  with TrackNullsParam with NumericBucketizerMetadata
  with MapPivotParams with CleanTextMapFun
  with AllowLabelAsInput[OPVector] {

  def fitFn(dataset: Dataset[(Option[Double], Map[String, N])]): BinaryModel[RealNN, I2, OPVector] = {
    import dataset.sparkSession.implicits._
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = false

    // drop the empty map values & clean map keys if needed
    val ds = dataset.filter(_._2.nonEmpty).map { case (label, map) =>
      label -> filterKeys[N](map, shouldCleanKey = shouldCleanKeys, shouldCleanValue = shouldCleanValues)
    }.persist()

    val computedSplits: Array[(String, Splits)] = if (ds.isEmpty) {
      log.info("Skip bucketizing empty numeric map '{}' feature", in2.name)
      Array.empty[(String, Splits)]
    } else {
      // Collect all unique map keys and sort them
      val uniqueKeys: Seq[String] =
        ds.map { case (_, map) => map.keys.toSeq }
          .reduce((l, r) => (l ++ r).distinct)
          .distinct.sorted

      // Compute splits for each collected key in parallel
      uniqueKeys.par.map { k =>
        val data: Dataset[(Double, Double)] =
          ds.filter(_._2.contains(k))
            .map { case (label, map) => label.get -> nev.toDouble(map(k)) }
        k -> computeSplits(data, featureName = s"${in2.name}[$k]")
      }.toArray
    }
    ds.unpersist()

    val meta = makeMetadata(computedSplits)
    setMetadata(meta)

    new DecisionTreeNumericMapBucketizerModel[I2](
      shouldSplitByKey = computedSplits.map { case (k, s) => k -> s.shouldSplit }.toMap,
      splitsByKey = computedSplits.map { case (k, s) => k -> s.splits }.toMap,
      trackNulls = $(trackNulls),
      trackInvalid = $(trackInvalid),
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      operationName = operationName,
      uid = uid
    )
  }

  private def makeMetadata(allSplits: Array[(String, Splits)]): Metadata = {
    val cols: Array[Array[OpVectorColumnMetadata]] = allSplits.map { case (key, split) =>
      makeVectorColumnMetadata(
        input = in2,
        grouping = Some(key),
        bucketLabels = split.bucketLabels,
        trackNulls = $(trackNulls),
        trackInvalid = split.shouldSplit && $(trackInvalid)
      )
    }
    vectorMetadataFromInputFeatures.withColumns(cols.flatten).toMetadata
  }

}

final class DecisionTreeNumericMapBucketizerModel[I2 <: OPMap[_]] private[op]
(
  val shouldSplitByKey: Map[String, Boolean],
  val splitsByKey: Map[String, Array[Double]],
  val trackNulls: Boolean,
  val trackInvalid: Boolean,
  val shouldCleanKeys: Boolean,
  val shouldCleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti2: TypeTag[I2])
  extends BinaryModel[RealNN, I2, OPVector](operationName = operationName, uid = uid)
  with CleanTextMapFun with AllowLabelAsInput[OPVector] {

  private val keys = shouldSplitByKey.keys.toArray.sorted
  private val keyIndices = keys.indices

  def transformFn: (RealNN, I2) => OPVector = (_, input) => {
    val inDoubleMap: Map[String, Double] = input.value.mapValues {
      case v: Double => v
      case l: Long => l.toDouble
      case b: Boolean => if (b) 1.0 else 0.0
      case v => throw new RuntimeException(s"Value '$v' cannot be converted to Double")
    }
    val cleanedInputMap = cleanMap(inDoubleMap, shouldCleanKey = shouldCleanKeys, shouldCleanValue = shouldCleanValues)
    val vectors =
      for {i <- keyIndices; k = keys(i)} yield {
        NumericBucketizer.bucketize(
          shouldSplit = shouldSplitByKey.get(k).contains(true),
          splits = splitsByKey(k),
          trackNulls = trackNulls,
          trackInvalid = trackInvalid,
          splitInclusion = DecisionTreeNumericBucketizer.Inclusion,
          input = cleanedInputMap.get(k)
        )
      }
    combine(vectors).toOPVector
  }

}
