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

package com.salesforce.op.testkit

import com.salesforce.op.features.types.{Currency, Percent, Real, RealNN}
import com.salesforce.op.test.TestCommon
import com.salesforce.op.testkit.RandomReal._
import org.apache.spark.ml.linalg._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

// [W-4293013] - more random vectors

@RunWith(classOf[JUnitRunner])
class RandomVectorTest extends FlatSpec with TestCommon {
  val numTries = 1000000

  Spec[RandomVector] should "produce dense vectors" in {
    val vectors = RandomVector.dense(RandomReal.uniform[Real](-1.0, 1.0), 3)
    check(vectors, predicate = (v: Vector) => {
      v.size == 3 && v.toArray.forall(x => 1.0 > math.abs(x))
    })
  }

  it should "produce sparse vectors" in {
    val reals = RandomReal.poisson[Real](4.0).withProbabilityOfEmpty(0.4)
    val vectors = RandomVector.sparse(reals, 3)
    check(vectors,
      predicate = _.size == 3,
      expected = List(
        List(4.0, 0.0, 0.0),
        List(0.0, 7.0, 1.0),
        List(4.0, 5.0, 3.0),
        List(5.0, 0.0, 7.0),
        List(4.0, 2.0, 3.0)
      )
    )
  }

  it should "Give values distributed in the cloud by the law" in {
    val center = new DenseVector(Array(4.0, 3.0))
    val matrix = new DenseMatrix(2, 2, Array(8, 0.6, 0.6, 0.8))
    val sut = RandomVector.normal(center, matrix)

    sut reset 77
    val found = sut.next
    sut reset 77
    val foundAfterReseed = sut.next
    withClue(s"generator reset did not work for $sut: $found/$foundAfterReseed") {
      foundAfterReseed shouldBe found
    }

    check(sut, predicate = _ => true, expected = List(
      List(2.2996685228637697, 4.020626621218229),
      List(7.0239295306677665, 4.64383918464643),
      List(2.2776269335796417, 2.506848417731993),
      List(-0.746412841570697, 3.813613151074187)
    ) )
  }
  it should "Give ones and zeroes with given probability" in {
    val sut = RandomVector.binary(4, probabilityOfOne = 0.5)

    sut reset 42
    val actualSum = sut take numTries map (v => v.value.toArray.sum) sum

    withClue (s"Got $actualSum ones out of $numTries") {
      math.abs(actualSum - numTries/2*4) < numTries / 100 shouldBe true
    }

    check(sut, predicate = v => v.size == 4 && v.toArray.forall(x => math.abs(x - 0.5) == 0.5),
      expected = List(
        List(1.0, 1.0, 1.0, 1.0),
        List(1.0, 0.0, 0.0, 0.0),
        List(1.0, 1.0, 1.0, 0.0),
        List(0.0, 1.0, 0.0, 1.0),
        List(0.0, 0.0, 1.0, 0.0)
      ))
  }

  it should "Give ones and zeroes with given probabilities" in {
    val probs = Array(0.5, 0.1, 0.9, 0.3)

    def plus(v1: Vector, v2: Vector): Vector = new DenseVector (
      v1.toArray zip v2.toArray map { case (x, y) => x + y }
    )

    def minus(v1: Vector, v2: Vector): Vector = new DenseVector (
      v1.toArray zip v2.toArray map { case (x, y) => x - y }
    )

    def dist(v1: Vector, v2: Vector): Double = {
      val diff = minus(v1, v2)
      math.sqrt(diff.toArray map (x => x * x) sum)
    }

    val sut = RandomVector.binary(probs)

    val expected = new DenseVector(probs map (numTries *))

    sut reset 42
    val vectors = sut limit numTries map (v => v.value)
    val actualSum = (Vectors.zeros(4) /: vectors)(plus)

    val diff = minus(actualSum, expected)

    withClue (s"Got $actualSum ones out of $numTries; expected $expected") {
      dist(actualSum, expected) < numTries / 100 shouldBe true
    }

    check(sut, predicate = v => v.size == 4 && v.toArray.forall(x => math.abs(x - 0.5) == 0.5),
      expected = List(
        List(1.0, 1.0, 1.0, 1.0),
        List(1.0, 0.0, 1.0, 0.0),
        List(1.0, 0.0, 1.0, 0.0),
        List(0.0, 0.0, 1.0, 1.0),
        List(0.0, 0.0, 1.0, 0.0)
      ))
  }

  private val rngSeed = 1414235000

  private def check(
    sut: RandomVector,
    predicate: Vector => Boolean,
    expected: Seq[List[Double]] = Nil
  ) = {
    sut reset rngSeed

    val found = sut.next
    sut reset rngSeed
    val foundAfterReseed = sut.next
    withClue(s"generator reset did not work for $sut") {
      foundAfterReseed shouldBe found
    }
    sut reset rngSeed

    val numberOfOutliers = sut limit numTries count (opv => !predicate(opv.value))

    numberOfOutliers should be < (numTries / 1000)
    val samples = sut limit expected.length
    for { i <- expected.indices } {
      withClue(s"At $i") {
        samples(i).value.toArray.toList shouldBe expected(i)
      }
    }
  }
}
