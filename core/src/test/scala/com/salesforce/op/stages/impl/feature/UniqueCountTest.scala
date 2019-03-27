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

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.testkit.{RandomMap, RandomReal, RandomVector}
import com.twitter.algebird.{Monoid, Semigroup}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class UniqueCountTest extends FlatSpec with TestSparkContext with UniqueCountFun {
  import spark.implicits._

  implicit val kryo = new KryoSerializer(spark.sparkContext.getConf)

  Spec[UniqueCountFun] should "count uniques" in {
    val (numRows, numCols, bits) = (1000, 10, 12)
    val data = createData(numRows, numCols)
    val (uniqueCounts, total) = countUniques[Double](data, size = numCols, bits = bits)
    total shouldBe data.count()
    val expected = expectedCountUniques(data)
    uniqueCounts.map(_.estimatedSize.toInt) shouldBe expected
  }

  it should "count uniques on empty data" in {
    val (numRows, numCols, bits) = (0, 10, 12)
    val data = createData(numRows, numCols)
    val (uniqueCounts, total) = countUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    uniqueCounts.size shouldBe numCols
    uniqueCounts.foreach(_.estimatedSize shouldBe 0.0)

  }

  it should "count uniques in maps" in {
    val (numRows, numCols, bits) = (1000, 10, 12)
    val data = createMapData(numRows, numCols)
    val (uniqueCounts, total) = countMapUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    val expected = expectedCountMapUniques(data)
    uniqueCounts.flatMap(_.map { case (_, v) => v.estimatedSize.toInt }) shouldBe expected
  }

  it should "count uniques in maps on empty data" in {
    val (numRows, numCols, bits) = (0, 10, 12)
    val data = createMapData(numRows, numCols)
    val (uniqueCounts, total) = countMapUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    uniqueCounts.size shouldBe numCols
    uniqueCounts.foreach(_.size shouldBe 0)
  }


  private def createData(nRows: Int, nCols: Int): Dataset[Seq[Double]] = {
    RandomVector.dense(RandomReal.poisson(10), nCols)
      .limit(nRows)
      .map(_.value.toArray.toSeq)
      .toDS()
  }

  private def createMapData(nRows: Int, nCols: Int): Dataset[Seq[Map[String, Double]]] = {
    val poisson = RandomReal.poisson[Real](10)
    ((0 until nCols).map(j =>
      RandomMap.ofReals[Real, RealMap](poisson, 0, 10)
        .limit(nRows)
        .map(_.value)
    ).transpose: Seq[Seq[Map[String, Double]]]).toDS()
  }

  private def expectedCountUniques(data: Dataset[Seq[Double]]): Seq[Int] = {
    data.rdd
      .map(_.map(v => Map(v -> 1L)))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => MapTestMonoids.mapDLMonoid.plus(m1, m2) })
      .map(_.size)
  }

  private def expectedCountMapUniques(data: Dataset[Seq[Map[String, Double]]]): Seq[Int] = {
    data.rdd
      .map(_.map(_.flatMap { case (k, v) => Map(k -> Map(v -> 1L)) }))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => MapTestMonoids.mapMDLMonoid.plus(m1, m2) })
      .flatMap(_.map(_._2.size))
  }
}

private object MapTestMonoids {
  val longSG = Semigroup.from[Long](_ + _)
  val mapDLMonoid = Monoid.mapMonoid[Double, Long](longSG)  // we use Semigroup here to avoid map keys removal
  val mapMDLMonoid = Monoid.mapMonoid[String, Map[Double, Long]](mapDLMonoid)
}
