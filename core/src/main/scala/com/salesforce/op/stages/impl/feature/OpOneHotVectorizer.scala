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
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.twitter.algebird.{HLL, HyperLogLogMonoid}
import com.twitter.algebird.Operators._
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamValidators, Params}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Dataset, Encoder}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag


/**
 * Converts a sequence of features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
abstract class OpOneHotVectorizer[T <: FeatureType]
(
  operationName: String,
  uid: String = UID[OpOneHotVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = operationName, uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with OneHotFun with MaxPctCardinalityParams {

  protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]]

  protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String): SequenceModel[T, OPVector]

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanText = $(cleanText)
    val shouldTrackNulls = $(trackNulls)

    val maxPctCard = $(maxPctCardinality)
    val rdd: RDD[Seq[Map[String, Int]]] = convertToSeqOfMaps(dataset)

    val finalRDD =
      if (maxPctCard == 1.0) rdd
      else {
        implicit val classTag: ClassTag[T#Value] = ReflectionUtils.classTagForWeakTypeTag[T#Value]
        implicit val kryo = new KryoSerializer(dataset.sparkSession.sparkContext.getConf)

        val (uniqueCounts, n) = countUniques(dataset, size = inN.length, bits = $(hllBits))

        // Throw out values above max percent cardinality
        val percentFilter = uniqueCounts.map(_.estimatedSize / n < maxPctCard)
        rdd.map(_.zip(percentFilter).collect { case (v, true) => v })
      }

    val empty = Seq.fill(inN.length)(Map.empty[String, Int])
    val countOccurrences: Seq[Map[String, Int]] =
      finalRDD.fold(empty)((a, b) => a.zip(b).map { case (m1, m2) => m1 + m2 })

    // Top K values for each categorical input
    val numToKeep = $(topK)
    val minSup = $(minSupport)
    val topValues: Seq[Seq[String]] =
      countOccurrences.map(_.toSeq.filter(_._2 >= minSup).sortBy(v => -v._2 -> v._1).take(numToKeep).map(_._1))

    // build metadata describing output
    val unseen = Option($(unseenName))
    val vecMetadata = makeVectorMetadata(
      shouldTrackNulls = shouldTrackNulls,
      unseen = unseen,
      topValues = topValues,
      outputName = getOutputFeatureName,
      features = getTransientFeatures(),
      stageName = stageName
    )
    setMetadata(vecMetadata.toMetadata)

    makeModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid
    )
  }
}

object OpOneHotVectorizer {
  /**
   * Default value for max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
   */
  val MaxPctCardinality = 1.0

  /**
   * Number of bits used for hashing in HyperLogLog (HLL). Error is about 1.04/sqrt(2^{bits}).
   * Default is 12 bits for 1% error which means each HLL instance is about 2^{12} = 4kb per instance.
   */
  val HLLBits = 12
}

abstract class OpOneHotVectorizerModel[T <: FeatureType]
(
  val topValues: Seq[Seq[String]],
  val shouldCleanText: Boolean,
  val shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with CleanTextFun with OneHotModelFun[T] {

  def transformFn: Seq[T] => OPVector = pivotFn(topValues, shouldCleanText, shouldTrackNulls)

}

/**
 * Converts a sequence of OpSet features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
class OpSetVectorizer[T <: OPSet[_]]
(
  operationName: String = "vecSet",
  uid: String = UID[OpSetVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OpOneHotVectorizer[T](operationName = operationName, uid = uid){

  override protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]] = {
    val shouldCleanText = $(cleanText)

    dataset.rdd.map(_.map(cat =>
      cat.map(v => cleanTextFn(v.toString, shouldCleanText) -> 1).toMap
    ))
  }

  override protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String) =
    new OpSetVectorizerModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid
    ).setCleanTextParams($(cleanTextParams))

}

final class OpSetVectorizerModel[T <: OPSet[_]] private[op]
(
  topValues: Seq[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OpOneHotVectorizerModel[T](topValues, shouldCleanText, shouldTrackNulls, operationName, uid) {
  override protected def convertToSet(in: T): Set[_] = in.value.toSet
}


/**
 * Converts a sequence of Text features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
class OpTextPivotVectorizer[T <: Text]
(
  operationName: String = "pivotText",
  uid: String = UID[OpSetVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OpOneHotVectorizer[T](operationName = operationName, uid = uid){

  override protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]] = {
    val shouldCleanText = $(cleanText)

    dataset.rdd.map(_.map(cat =>
      cat.map(v => cleanTextFn(v, shouldCleanText) -> 1).toMap
    ))
  }

  override protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String) =
    new OpTextPivotVectorizerModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid
    ).setCleanTextParams($(cleanTextParams))
}

final class OpTextPivotVectorizerModel[T <: Text] private[op]
(
  topValues: Seq[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OpOneHotVectorizerModel[T](topValues, shouldCleanText, shouldTrackNulls, operationName, uid) {
  override protected def convertToSet(in: T): Set[_] = in.value.toSet
}


trait MaxPctCardinalityParams extends Params {
  final val maxPctCardinality = new DoubleParam(
    parent = this, name = "maxPctCardinality",
    doc = "max percentage of distinct values a categorical feature can have",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0)
  )
  final def setMaxPctCardinality(v: Double): this.type = set(maxPctCardinality, v)
  final def getMaxPctCardinality: Double = $(maxPctCardinality)
  setDefault(maxPctCardinality -> OpOneHotVectorizer.MaxPctCardinality)

  final val hllBits = new IntParam(
    parent = this, name = "hllBits", doc =
      "Number of bits used for hashing in HyperLogLog (HLL). Error is about 1.04/sqrt(2^{bits})." +
        " Default is 12 bits for 1% error which means each HLL instance is about 2^{12} = 4kb per instance.",
    isValid = ParamValidators.inRange(lowerBound = 4, upperBound = 31)
  )
  final def setHLLBits(value: Int): this.type = set(hllBits, value)
  final def getHLLBits: Int = $(hllBits)
  setDefault(hllBits -> OpOneHotVectorizer.HLLBits)
}


private[op] trait MaxPctCardinalityFun extends UniqueCountFun {

  /**
   * Throw out values above max percent cardinality as follows:
   *
   * 1. count unique values of each of the sequence & map key components in the dataset using HyperLogLog [[HLL]]
   * 2. count total number of rows
   * 3. compute percent filters for each of the sequence & map key components
   * 4. throw out values above max percent cardinality
   *
   * @param dataset           dataset to count unique values
   * @param maxPctCardinality max percentage of distinct values a categorical feature can have
   * @param size              size of each sequence component
   * @param bits              number of bits for HyperLogLog [[HLL]]
   * @param tt                type tag of V - needed by kryo
   * @param enc               values encoder for dataset
   * @tparam V value type
   * @return
   */
  def filterByMaxCardinality[V]
  (
    dataset: Dataset[Seq[Map[String, V]]],
    maxPctCardinality: Double,
    size: Int,
    bits: Int
  )(implicit tt: TypeTag[V], enc: Encoder[Seq[Map[String, V]]]): Dataset[Seq[Map[String, V]]] = {
    require(maxPctCardinality >= 0.0 && maxPctCardinality <= 1.0, "maxPctCardinality must be in range [0.0, 1.0]")

    if (maxPctCardinality == 1.0) dataset
    else {
      implicit val classTag: ClassTag[V] = ReflectionUtils.classTagForWeakTypeTag[V]
      implicit val kryo = new KryoSerializer(dataset.sparkSession.sparkContext.getConf)

      // Count unique values of each of the sequence & map key components in the dataset using HyperLogLog [[HLL]]
      val (uniqueCounts, n) = countMapUniques(dataset, size = size, bits = bits)

      // Throw out values above max percent cardinality
      val percentFilters = uniqueCounts.map(_.map { case (k, v) => (k, v.estimatedSize / n < maxPctCardinality) })
      dataset.map(_.zip(percentFilters).collect { case (m, percentFilter) =>
        m.filter { case (k, _) => percentFilter.getOrElse(k, true) }
      })
    }
  }

}

/**
 * Provides functions to count cardinality of data using HyperLogLog [[HLL]]
 */
private[op] trait UniqueCountFun {

  /**
   * Count unique values of each of the sequence components in the dataset using HyperLogLog [[HLL]]
   *
   * @param dataset dataset to count unique values
   * @param size    size of each sequence component
   * @param bits    number of bits for HyperLogLog [[HLL]]
   * @param kryo    kryo serializer to serialize [[V]] value into array of bytes
   * @param ct      class tag of V - needed by kryo
   * @tparam V value type
   * @return HyperLogLog [[HLL]] of unique values count for each of the sequence components and total rows count
   */
  def countUniques[V](dataset: Dataset[Seq[V]], size: Int, bits: Int)
    (implicit kryo: KryoSerializer, ct: ClassTag[V]): (Seq[HLL], Long) = {
    val key = "k" // lift values into map with a single key
    val rdd = dataset.rdd.map(seq => seq.map(v => Map(key -> v)))
    val (counts, total) = countMapUniques(rdd, size = size, bits = bits)
    val zero = new HyperLogLogMonoid(bits).zero
    counts.map(_.getOrElse(key, zero)) -> total
  }

  /**
   * Count unique values of each of the sequence & map key components in the dataset using HyperLogLog [[HLL]]
   *
   * @param dataset dataset to count unique values
   * @param size    size of each sequence component
   * @param bits    number of bits for HyperLogLog [[HLL]]
   * @param kryo    kryo serializer to serialize [[V]] value into array of bytes
   * @param ct      class tag of V - needed by kryo
   * @tparam V value type
   * @return HyperLogLog [[HLL]] of unique values count for each of the sequence components and total rows count
   */
  def countMapUniques[V]
  (
    dataset: Dataset[Seq[Map[String, V]]],
    size: Int,
    bits: Int
  )(implicit kryo: KryoSerializer, ct: ClassTag[V]): (Seq[Map[String, HLL]], Long) = {
    countMapUniques(dataset.rdd, size = size, bits = bits)
  }

  private def countMapUniques[V]
  (
    rdd: RDD[Seq[Map[String, V]]],
    size: Int,
    bits: Int
  )(implicit kryo: KryoSerializer, ct: ClassTag[V]): (Seq[Map[String, HLL]], Long) = {
    implicit val hllMonoid = new HyperLogLogMonoid(bits)
    val hlls = rdd.mapPartitions { it =>
      val ks = kryo.newInstance() // reuse the same kryo instance for the partition
      it.map(_.map(_.map { case (k, v) => (k, hllMonoid.create(ks.serialize(v).array())) }) -> 1L)
    }
    val zero = Seq.fill(size)(Map.empty[String, HLL]) -> 0L
    val countMapUniques = hlls.fold(zero) { case ((a, c1), (b, c2)) =>
      a.zip(b).map { case (m1, m2) => m1 + m2 } -> (c1 + c2)
    }
    countMapUniques
  }

}

/**
 * One Hot Functionality
 */
private[op] trait OneHotFun extends UniqueCountFun {

  protected def makeVectorColumnMetadata(
    shouldTrackNulls: Boolean, unseen: Option[String], topValues: Seq[Seq[String]], features: Array[TransientFeature]
  ): Array[OpVectorColumnMetadata] = {
    for {
      (parentFeature, values) <- features.zip(topValues)
      parentFeatureType = parentFeature.typeName
      // Append other/null indicators for each input (view here to avoid copying the array when appending the string)
      value <-
        if (shouldTrackNulls) values.map(Option(_)).view ++ Array(unseen, Option(TransmogrifierDefaults.NullString))
        else values.map(Option(_)).view :+ unseen
    } yield parentFeature.toColumnMetaData(true).copy(indicatorValue = value)
  }

  protected def makeVectorMetadata(
    shouldTrackNulls: Boolean, unseen: Option[String], topValues: Seq[Seq[String]], outputName: String,
    features: Array[TransientFeature], stageName: String
  ): OpVectorMetadata = {
    val columns = makeVectorColumnMetadata(shouldTrackNulls, unseen, topValues, features)
    OpVectorMetadata(outputName, columns, Transmogrifier.inputFeaturesToHistory(features, stageName))
  }
}

/**
 * One Hot Model Functionality
 */
private[op] trait OneHotModelFun[T <: FeatureType] extends CleanTextFun {
  protected def convertToSet(in: T): Set[_]

  protected def pivotFn(
    topValues: Seq[Seq[String]], shouldCleanText: Boolean, shouldTrackNulls: Boolean
  ): Seq[T] => OPVector = row => {
    // Combine top values for each feature with categorical feature
    val eachPivoted = row.zip(topValues).map { case (cat, top) =>
      val theseCat = convertToSet(cat)
        .groupBy(v => cleanTextFn(v.toString, shouldCleanText))
        .map { case (k, v) => k -> v.size }
      val topPresent = top.zipWithIndex.collect { case (c, i) if theseCat.contains(c) => (i, theseCat(c).toDouble) }
      val notPresent = theseCat.keySet.diff(top.toSet).toSeq
      val notPresentVal = notPresent.map(theseCat).sum.toDouble
      val nullVal = if (theseCat.isEmpty) 1.0 else 0.0
      // Append the other and null entries to the vector (note topPresent is sparse, so use top.length as proxy for K)
      if (shouldTrackNulls) topPresent ++ Array((top.length, notPresentVal), (top.length + 1, nullVal))
      else topPresent :+ (top.length, notPresentVal)
    }

    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted)
    val vector = makeSparseVector(reindexed)
    vector.toOPVector
  }
}
