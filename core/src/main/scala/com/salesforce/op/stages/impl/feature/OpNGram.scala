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
import org.apache.spark.ml.feature.NGram

/**
 * Wrapper for [[org.apache.spark.ml.feature.NGram]]
 *
 * A feature transformer that converts the input array of strings into an array of n-grams. Null
 * values in the input array are ignored.
 * It returns an array of n-grams where each n-gram is represented by a space-separated string of
 * words.
 *
 * When the input is empty, an empty array is returned.
 * When the input array length is less than n (number of elements per n-gram), no n-grams are
 * returned.
 *
 * @see [[NGram]] for more info
 */
class OpNGram(uid: String = UID[NGram])
  extends OpTransformerWrapper[TextList, TextList, NGram](transformer = new NGram(), uid = uid) {

  /**
   * Minimum n-gram length, greater than or equal to 1.
   * Default: 2, bigram features
   */
  def setN(value: Int): this.type = {
    getSparkMlStage().get.setN(value)
    this
  }

}
