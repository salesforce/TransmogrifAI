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

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.spark.RichVector._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OPVectorTest extends FlatSpec with TestCommon {

  val vectors = Seq(
    Vectors.sparse(4, Array(0, 3), Array(1.0, 1.0)).toOPVector,
    Vectors.dense(Array(2.0, 3.0, 4.0)).toOPVector,
    // Purposely added a very large sparse vector to verify the efficiency
    Vectors.sparse(100000000, Array(1), Array(777.0)).toOPVector
  )

  Spec[OPVector] should "be empty" in {
    val zero = Vectors.zeros(0)
    new OPVector(zero).isEmpty shouldBe true
    new OPVector(zero).nonEmpty shouldBe false
    zero.toOPVector shouldBe a[OPVector]
  }

  it should "error on size mismatch" in {
    val ones = Array.fill(vectors.size)(Vectors.sparse(1, Array(0), Array(1.0)).toOPVector)
    for {
      (v1, v2) <- vectors.zip(ones)
      res <- Seq(() => v1 + v2, () => v1 - v2, () => v1 dot v2)
    } intercept[IllegalArgumentException](res()).getMessage should {
      (startWith("requirement failed: Vectors must") and include("same length")) or
        (startWith("requirement failed:") and include("Vectors with non-matching sizes"))
    }
  }

  it should "compare values" in {
    val zero = Vectors.zeros(0)
    new OPVector(zero) shouldBe new OPVector(zero)
    new OPVector(zero).value shouldBe zero

    Vectors.dense(Array(1.0, 2.0)).toOPVector shouldBe Vectors.dense(Array(1.0, 2.0)).toOPVector
    Vectors.sparse(5, Array(3, 4), Array(1.0, 2.0)).toOPVector shouldBe
      Vectors.sparse(5, Array(3, 4), Array(1.0, 2.0)).toOPVector
    Vectors.dense(Array(1.0, 2.0)).toOPVector should not be Vectors.dense(Array(2.0, 2.0)).toOPVector
    new OPVector(Vectors.dense(Array(1.0, 2.0))) should not be Vectors.dense(Array(2.0, 2.0)).toOPVector
    OPVector.empty shouldBe new OPVector(zero)
  }

  it should "'+' add" in {
    for {(v1, v2) <- vectors.zip(vectors)} {
      (v1 + v2) shouldBe (v1.value + v2.value).toOPVector
    }
  }

  it should "'-' subtract" in {
    for {(v1, v2) <- vectors.zip(vectors)} {
      (v1 - v2) shouldBe (v1.value - v2.value).toOPVector
    }
  }

  it should "compute dot product" in {
    for {(v1, v2) <- vectors.zip(vectors)} {
      (v1 dot v2) shouldBe (v1.value dot v2.value)
    }
  }

  it should "combine" in {
    for {(v1, v2) <- vectors.zip(vectors)} {
      v1.combine(v2) shouldBe v1.value.combine(v2.value).toOPVector
      v1.combine(v2, v2, v1) shouldBe v1.value.combine(v2.value, v2.value, v1.value).toOPVector
    }
  }

}
