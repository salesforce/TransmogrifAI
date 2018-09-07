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

package com.salesforce.op.utils.stats

import com.salesforce.op.test.TestCommon
import org.apache.spark.mllib.linalg.DenseMatrix
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

@RunWith(classOf[JUnitRunner])
class OpStatisticsPropertyTest extends PropSpec with TestCommon with PropertyChecks {

  val genInt = Gen.posNum[Int]
  private def genArray(n: Int) = Gen.containerOfN[Array, Int](n, genInt)

  val genMatrix = for {
    rowSize <- Gen.choose(1, 13)
    colSize <- Gen.choose(1, 13)
    size = rowSize * colSize
    array <- genArray(size)
  } yield {
    new DenseMatrix(rowSize, colSize, array.map(_.toDouble))
  }

  property("cramerV function should produce results in expected ranges") {
    forAll(genMatrix) { (matrix: DenseMatrix) =>
      val res = OpStatistics.chiSquaredTest(matrix).cramersV
      if (matrix.numRows > 1 && matrix.numCols > 1) {
        res >= 0 shouldBe true
        res <= 1 shouldBe true
      } else {
        res.isNaN shouldBe true
      }
    }
  }
}
