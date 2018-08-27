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
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid._
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import enumeratum._
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}

import scala.reflect.runtime.universe.TypeTag

/**
 * OP wrapper for [[org.apache.spark.ml.feature.StringIndexer]]
 *
 * NOTE THAT THIS CLASS EITHER FILTERS OUT OR THROWS AN ERROR IF PREVIOUSLY UNSEEN VALUES APPEAR
 *
 * A label indexer that maps a text column of labels to an ML feature of label indices.
 * The indices are in [0, numLabels), ordered by label frequencies.
 * So the most frequent label gets index 0.
 *
 * @see [[OpIndexToString]] for the inverse transformation
 */
class OpStringIndexer[T <: Text]
(
  uid: String = UID[OpStringIndexer[T]]
)(implicit tti: TypeTag[T])
  extends OpEstimatorWrapper[T, RealNN, StringIndexer, StringIndexerModel](estimator = new StringIndexer(), uid = uid) {

  /**
   * How to handle invalid entries. See [[StringIndexer.handleInvalid]] for more details.
   *
   * @param value StringIndexerHandleInvalid
   * @return this stage
   */
  def setHandleInvalid(value: StringIndexerHandleInvalid): this.type = {
    assert(Seq(Skip, Error, Keep).contains(value), "OpStringIndexer only supports Skip, Error, and Keep for handle invalid")
    getSparkMlStage().get.setHandleInvalid(value.entryName.toLowerCase)
    this
  }
}

sealed trait StringIndexerHandleInvalid extends EnumEntry with Serializable

object StringIndexerHandleInvalid extends Enum[StringIndexerHandleInvalid] {
  val values = findValues
  case object Skip extends StringIndexerHandleInvalid
  case object Error extends StringIndexerHandleInvalid
  case object Keep extends StringIndexerHandleInvalid
  case object NoFilter extends StringIndexerHandleInvalid
}
