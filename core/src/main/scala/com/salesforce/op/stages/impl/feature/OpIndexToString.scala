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
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import enumeratum._
import org.apache.spark.ml.feature.IndexToString

/**
 * OP wrapper for [[org.apache.spark.ml.feature.IndexToString]]
 *
 * NOTE THAT THIS CLASS EITHER FILTERS OUT OR THROWS AN ERROR IF PREVIOUSLY UNSEEN VALUES APPEAR
 *
 * A transformer that maps a feature of indices back to a new feature of corresponding text values.
 * The index-string mapping is either from the ML attributes of the input feature,
 * or from user-supplied labels (which take precedence over ML attributes).
 *
 * @see [[OpStringIndexer]] for converting text into indices
 */
class OpIndexToString(uid: String = UID[OpIndexToString])
  extends OpTransformerWrapper[RealNN, Text, IndexToString](
    transformer = new IndexToString(), uid = uid
  ) {

  /**
   * Optional array of labels specifying index-string mapping.
   * If not provided or if empty, then metadata from input feature is used instead.
   *
   * @param value array of labels
   * @return
   */
  def setLabels(value: Array[String]): this.type = {
    getSparkMlStage().get.setLabels(value)
    this
  }

  /**
   * Array of labels
   *
   * @return Array of labels
   */
  def getLabels: Array[String] = getSparkMlStage().get.getLabels
}


sealed trait IndexToStringHandleInvalid extends EnumEntry with Serializable

object IndexToStringHandleInvalid extends Enum[IndexToStringHandleInvalid] {
  val values = findValues
  case object NoFilter extends IndexToStringHandleInvalid
  case object Error extends IndexToStringHandleInvalid
}
