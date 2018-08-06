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

package com.salesforce.op.stages.impl

import org.apache.spark.ml.param.ParamMap
import org.scalatest.{Assertions, Matchers}


trait CompareParamGrid extends Matchers with Assertions {

  /**
   * Compare two params grids
   */
  def gridCompare(g1: Array[ParamMap], g2: Array[ParamMap]): Unit = {
    val g1values = g1.toSet[ParamMap].map(_.toSeq.toSet)
    val g2values = g2.toSet[ParamMap].map(_.toSeq.toSet)
    matchTwoSets(g1values, g2values)
  }

  private def matchTwoSets[T](actual: Set[T], expected: Set[T]): Unit = {
    def stringify(set: Set[T]): String = {
      val list = set.toList
      val chunk = list take 10
      val strings = chunk.map(_.toString).sorted
      if (list.size > chunk.size) strings.mkString else strings.mkString + ",..."
    }
    val missing = stringify(expected -- actual)
    val extra = stringify(actual -- expected)
    withClue(s"Missing:\n $missing\nExtra:\n$extra") {
      actual shouldBe expected
    }
  }
}
