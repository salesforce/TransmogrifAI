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

package com.salesforce.op.filters

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class HistogramTest extends FlatSpec with TestCommon {

  Spec(classOf[Histogram]) should "produce correct histogram distribution" in {
    // Follows example in appendix of original paper.
    val hist = new Histogram(5)
    hist.update(23, 19, 10, 16, 36)
    hist.getBins should contain theSameElementsAs Seq(23.0, 19.0, 10.0, 16.0, 36.0).map(_ -> 1L)
    hist.update(2)
    hist.getBins should contain theSameElementsAs Seq(2.0, 10.0, 23.0, 36.0).map(_ -> 1L) ++ Seq(17.5 -> 2L)
    hist.update(9)
    hist.getBins should contain theSameElementsAs Seq(2.0, 23.0, 36.0).map(_ -> 1L) ++ Seq(9.5, 17.5).map(_ -> 2L)

    val hist2 = new Histogram(5)
    hist.update(32, 30, 45)
    val mergedhist = hist.merge(hist2)
    mergedhist.getBins.map { case (point, count) =>
      (math.round(point * 100).toDouble / 100) -> count // Rounding out to 2 decimal places
    } should contain theSameElementsAs Seq(2.0 -> 1L, 9.5 -> 2L, 19.33 -> 3L, 32.67 -> 3L, 45 -> 1L)
  }
}
