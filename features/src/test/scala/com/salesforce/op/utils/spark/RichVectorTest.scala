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

package com.salesforce.op.utils.spark

import com.holdenkarau.spark.testing.RDDGenerator
import com.salesforce.op.features.types.ConcurrentCheck
import com.salesforce.op.test.TestSparkContext
import com.twitter.algebird.Monoid
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalactic.TolerantNumerics
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

import scala.concurrent.duration._


@RunWith(classOf[JUnitRunner])
class RichVectorTest extends PropSpec with PropertyChecks with TestSparkContext with ConcurrentCheck {

  import VectorGenerators._
  import com.salesforce.op.utils.spark.RichVector._

  lazy val sparseVevtorsRDDGen = RDDGenerator.genRDD[Vector](spark.sparkContext)(sparseVectorGen)

  property("Vectors should error on size mismatch") {
    forAll(sparseVectorGen) { sparse: SparseVector =>
      val wrongSize = Vectors.sparse(sparse.size + 1, Array(0), Array(1.0))
      val dense = sparse.toDense
      for {
        res <- Seq(
          () => sparse + wrongSize,
          () => sparse - wrongSize,
          () => dense + wrongSize,
          () => dense - wrongSize,
          () => dense + wrongSize.toDense,
          () => dense - wrongSize.toDense
        )
      } {
        intercept[IllegalArgumentException](res()).getMessage should {
          startWith("requirement failed: Vectors must") and include("same length")
        }
      }
    }
  }

  property("Vectors should '+' add correctly") {
    forAll(sparseVectorGen) { sparse: SparseVector =>
      val expected = sparse.toArray.map(_ * 2)
      val dense = sparse.toDense
      for {res <- Seq(sparse + sparse, dense + sparse, sparse + dense, dense + dense)} {
        res.size shouldBe sparse.size
        res.toArray should contain theSameElementsAs expected
      }
    }
  }

  property("Vectors should '-' subtract correctly") {
    forAll(sparseVectorGen) { sparse: SparseVector =>
      val dense = sparse.toDense
      for {res <- Seq(sparse - sparse, dense - sparse, sparse - dense, dense - dense)} {
        res.size shouldBe sparse.size
        res.toArray.foreach(_ shouldBe 0.0)
      }
    }
  }

  property("Vectors convert to breeze vectors correctly") {
    forAll(sparseVectorGen) { sparse: SparseVector =>
      val dense = sparse.toDense
      sparse.toBreeze.toArray should contain theSameElementsAs dense.toBreeze.toArray
    }
  }

  property("Sparse vectors should '+' add efficiently") {
    val sparseSize = 100000000
    val sparse = new SparseVector(sparseSize, Array(0, 1, sparseSize - 1), Array(-1.0, 1.0, 3.0))
    val expected = new SparseVector(sparseSize, Array(0, 1, sparseSize - 1), Array(-2.0, 2.0, 6.0))

    forAllConcurrentCheck[SparseVector](
      numThreads = 10, numInstancesPerThread = 100000, atMost = 10.seconds,
      table = Table[SparseVector]("sparseVectors", sparse),
      functionCheck = sparse => {
        val res = sparse + sparse
        res shouldBe a[SparseVector]
        res shouldEqual expected
      }
    )
  }

  property("Vectors add in reduce") {
    forAll(sparseVevtorsRDDGen) { rdd: RDD[Vector] =>
      if (!rdd.isEmpty()) {
        val tolerance = 1e-9 // we are loosing precision here, hence the tolerance
        implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(tolerance)

        val expected = rdd.map(_.toArray).reduce(Monoid.arrayMonoid[Double].plus)
        for {
          res <- Seq(
            () => rdd.reduce(_ + _),
            () => rdd.reduce(_.toDense + _),
            () => rdd.reduce(_ + _.toDense),
            () => rdd.reduce(_.toDense + _.toDense)
          )
          (v, exp) <- res().toArray.zip(expected)
        } v shouldEqual exp
      }
    }
  }

}

object VectorGenerators {

  val size = 100

  val sparseVectorGen = for {
    indices <- Gen.listOfN(size, Gen.choose(0, size - 1))
    values <- Gen.listOfN(size, Gen.choose(-100000.0, 100000.0).filter(!_.isNaN))
    idx = indices.distinct.sorted.toArray
  } yield new SparseVector(size, idx, values.toArray.take(idx.length))

}
