/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * A label indexer that maps a text column of labels to an ML feature of label indices.
 * The indices are in [0, numLabels), ordered by label frequencies.
 * So the most frequent label gets index 0.
 *
 * @see [[OpIndexToStringNoFilter]] for the inverse transformation
 */
class OpStringIndexerNoFilter[I <: Text]
(
  uid: String = UID[OpStringIndexerNoFilter[I]]
)(implicit tti: TypeTag[I], ttiv: TypeTag[I#Value])
  extends UnaryEstimator[I, RealNN](operationName = "str2idx", uid = uid) with SaveOthersParams {

  setDefault(unseenName, OpStringIndexerNoFilter.UnseenNameDefault)

  def fitFn(data: Dataset[I#Value]): UnaryModel[I, RealNN] = {
    val unseen = $(unseenName)
    val counts = data.rdd.countByValue()
    val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
    val otherPos = labels.length

    val cleanedLabels = labels.map(_.getOrElse("null")) :+ unseen
    val metadata = NominalAttribute.defaultAttr.withName(getOutputFeatureName).withValues(cleanedLabels).toMetadata()
    setMetadata(metadata)

    new OpStringIndexerNoFilterModel[I](labels, otherPos, operationName = operationName, uid = uid)
  }
}

final class OpStringIndexerNoFilterModel[I <: Text] private[op]
(
  val labels: Seq[Option[String]],
  val otherPos: Int,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I]) extends UnaryModel[I, RealNN](operationName = operationName, uid = uid) {

  private val labelsMap = labels.zipWithIndex.toMap
  def transformFn: I => RealNN = in => labelsMap.getOrElse(in.value, otherPos).toRealNN
}

object OpStringIndexerNoFilter {
  val UnseenNameDefault = "UnseenLabel"
}
