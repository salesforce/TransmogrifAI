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
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.param.{Param, StringArrayParam}

/**
 * A transformer that maps a feature of indices back to a new feature of corresponding text values.
 * The index-string mapping is either from the ML attributes of the input feature,
 * or from user-supplied labels (which take precedence over ML attributes).
 *
 * @see [[OpStringIndexerNoFilter]] for converting text into indices
 */
class OpIndexToStringNoFilter(uid: String = UID[OpIndexToStringNoFilter])
  extends UnaryTransformer[RealNN, Text](operationName = "idx2str", uid = uid) with SaveOthersParams {

  final val labels: StringArrayParam = new StringArrayParam(this, "labels",
    "Optional array of labels specifying index-string mapping." +
      " If not provided or if empty, then metadata from inputCol is used instead.")

  final def getLabels: Array[String] = $(labels)

  final def setLabels(labelsIn: Array[String]): this.type = set(labels, labelsIn)

  setDefault(unseenName, OpIndexToStringNoFilter.unseenDefault)

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN) => Text = {
    (input: RealNN) => {
      val inputColSchema = getInputSchema()(in1.name)
      // If the labels array is empty use column metadata
      val lbls = $(labels)
      val unseen = $(unseenName)
      val values = if (!isDefined(labels) || lbls.isEmpty) {
        Attribute.fromStructField(inputColSchema)
          .asInstanceOf[NominalAttribute].values.get
      } else {
        lbls
      }
      val idx = input.value.get.toInt
      if (0 <= idx && idx < values.length) {
        values(idx).toText
      } else {
        unseen.toText
      }
    }
  }
}

object OpIndexToStringNoFilter {
  val unseenDefault: String = "UnseenIndex"
}

