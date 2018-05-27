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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.feature.{OpIndexToStringNoFilter, SaveOthersParams}
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.sql.Dataset

import scala.util.{Failure, Success, Try}

/**
 * Estimator which takes response feature and predinction feature as inputs. It deindexes the pred by using response's
 * metadata
 *
 * Input 1 : response
 * Input 2 : pred feature
 *
 * @param uid
 */
class PredictionDeIndexer(uid: String = UID[PredictionDeIndexer])
  extends BinaryEstimator[RealNN, RealNN, Text](operationName = "idx2str", uid = uid) with SaveOthersParams {

  setDefault(unseenName, OpIndexToStringNoFilter.unseenDefault)

  /**
   * Function used to convert input to output
   */
  override def fitFn(dataset: Dataset[(Option[Double], Option[Double])]): BinaryModel[RealNN, RealNN, Text] = {
    val colSchema = getInputSchema()(in1.name)
    val labels: Array[String] = Try(Attribute.fromStructField(colSchema).asInstanceOf[NominalAttribute].values.get)
    match {
      case Success(l) => l
      case Failure(l) => throw new Error(s"The feature ${in1.name} does not contain" +
        s" any label/index mapping in its metadata")
    }

    new PredictionDeIndexerModel(labels, $(unseenName), operationName, uid)
  }
}

final class PredictionDeIndexerModel private[op]
(
  val labels: Array[String],
  val unseen: String,
  operationName: String,
  uid: String
) extends BinaryModel[RealNN, RealNN, Text](operationName = operationName, uid = uid) {

  def transformFn: (RealNN, RealNN) => Text = (response: RealNN, pred: RealNN) => {
    val idx = pred.value.get.toInt
    if (0 <= idx && idx < labels.length) labels(idx).toText
    else unseen.toText
  }

}
