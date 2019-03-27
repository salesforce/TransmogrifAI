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
import com.salesforce.op.testkit._
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
    val data = createDoubleData(numRows, numCols)
    val (uniqueCounts, total) = countUniques[Double](data, size = numCols, bits = bits)
    uniqueCounts.length shouldBe numCols
    total shouldBe data.count()
    val expected = expectedCountUniques(data)
    uniqueCounts.map(_.estimatedSize.toInt) shouldBe expected
  }

  it should "count uniques on empty data" in {
    val (numRows, numCols, bits) = (0, 10, 12)
    val data = createDoubleData(numRows, numCols)
    val (uniqueCounts, total) = countUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    uniqueCounts.size shouldBe numCols
    uniqueCounts.foreach(_.estimatedSize shouldBe 0.0)

  }

  it should "count uniques in maps" in {
    val (numRows, numCols, bits) = (1000, 10, 12)
    val data = createMapDoubleData(numRows, numCols)
    val (uniqueCounts, total) = countMapUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    val expected = expectedCountMapUniques(data)
    uniqueCounts.flatMap(_.map { case (_, v) => v.estimatedSize.toInt }) shouldBe expected
  }

  it should "count uniques in empty Maps" in {
    val (numRows, numCols, bits) = (1000, 10, 12)
    val data = Seq.fill(numRows)(Seq.fill(numCols)(Map.empty[String, Double])).toDS()
    val (uniqueCounts, total) = countMapUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    val expected = expectedCountMapUniques(data)
    uniqueCounts.flatMap(_.map { case (_, v) => v.estimatedSize.toInt }) shouldBe expected
  }

  it should "count uniques in maps on empty data" in {
    val (numRows, numCols, bits) = (0, 10, 12)
    val data = createMapDoubleData(numRows, numCols)
    val (uniqueCounts, total) = countMapUniques[Double](data, size = numCols, bits = bits)
    total shouldBe numRows
    uniqueCounts.size shouldBe numCols
    uniqueCounts.foreach(_.size shouldBe 0)
  }

  it should "count unique maps the same way as when counting uniques in columns" in {
    val (numRows, numCols, bits) = (1000, 10, 12)
    val data = createMapDoubleData(numRows, numCols)

    val exploded = explodeData(data, numCols)

    val explodedSize = exploded.first().length
    val countMap = countMapUniques(data, size = numCols, bits = bits)._1.flatMap(
      _.map { case (_, v) => v.estimatedSize.toInt})
    // Minus 1 because we don't count the None in the unique values
    val countExploded = countUniques(exploded, size = explodedSize, bits = bits)._1.map(_.estimatedSize.toInt - 1)


    countMap should contain theSameElementsAs countExploded

  }

  it should "count uniques Strings" in {
    val (numRows, numCols, bits) = (1000, 10, 18)
    val data = createStringData(numRows, numCols)
    val (uniqueCounts, total) = countUniques[String](data, size = numCols, bits = bits)
    total shouldBe data.count()
    val expected = expectedCountUniques(data)
    uniqueCounts.map(_.estimatedSize.toInt) shouldBe expected
  }

  it should "count uniques in Strings Maps" in {
    val (numRows, numCols, bits) = (1000, 10, 20)
    val data = createMapStringData(numRows, numCols)
    val (uniqueCounts, total) = countMapUniques[String](data, size = numCols, bits = bits)
    total shouldBe data.count()
    val expected = expectedCountMapUniques(data)
    uniqueCounts.flatMap(_.map { case (_, v) => v.estimatedSize.toInt }) shouldBe expected
  }

  private def createStringData(nRows: Int, nCols: Int): Dataset[Seq[String]] = {

    RandomList.ofTexts(texts = RandomText.cities, minLen = nCols, maxLen = nCols)
      .limit(nRows)
      .map(_.toArray.toSeq)
      .toDS()
  }

  private def createMapStringData(nRows: Int, nCols: Int): Dataset[Seq[Map[String, String]]] = {
    val texts = RandomText.cities
    ((0 until nCols).map(j =>
      RandomMap.of(texts, 0, 10)
        .limit(nRows)
        .map(_.value)
    ).transpose: Seq[Seq[Map[String, String]]]).toDS()
  }

  private def createDoubleData(nRows: Int, nCols: Int): Dataset[Seq[Double]] = {
    RandomVector.dense(RandomReal.poisson(10), nCols)
      .limit(nRows)
      .map(_.value.toArray.toSeq)
      .toDS()
  }

  private def createMapDoubleData(nRows: Int, nCols: Int): Dataset[Seq[Map[String, Double]]] = {
    val poisson = RandomReal.poisson[Real](10)
    ((0 until nCols).map(j =>
      RandomMap.ofReals[Real, RealMap](poisson, 0, 10)
        .limit(nRows)
        .map(_.value)
    ).transpose: Seq[Seq[Map[String, Double]]]).toDS()
  }

  private def explodeData(data: Dataset[Seq[Map[String, Double]]], numCols: Int): Dataset[Seq[Option[Double]]] = {
    val zero = Array.fill(numCols)(Set[String]())
    val keys = data.rdd.aggregate(zero)(
      seqOp = (u, row) => u.zipWithIndex.map { case (s, i) => s ++ row(i).keySet },
      combOp = (a, b) => a.zipWithIndex.map { case (s, i) => s ++ b(i) }
    )
    data.map(maps => maps.zipWithIndex.flatMap { case (m, i) =>
      keys(i).toSeq.map(key => m.get(key))
    })
  }

  private def expectedCountUniques[V](data: Dataset[Seq[V]]): Seq[Int] = {
    data.rdd
      .map(_.map(v => Map(v -> 1L)))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => MapTestMonoids.mapDLMonoid.plus(m1, m2) })
      .map(_.size)
  }

  private def expectedCountMapUniques[V](data: Dataset[Seq[Map[String, V]]]): Seq[Int] = {
    data.rdd
      .map(_.map(_.flatMap { case (k, v) => Map(k -> Map(v -> 1L)) }))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => MapTestMonoids.mapMDLMonoid.plus(m1, m2) })
      .flatMap(_.map(_._2.size))
  }
}

private object MapTestMonoids {
  val longSG = Semigroup.from[Long](_ + _)
  // we use Semigroup here to avoid map keys removal
  def mapDLMonoid[V]: Monoid[Map[V, Long]] = Monoid.mapMonoid[V, Long](longSG)
  def mapMDLMonoid[V]: Monoid[Map[String, Map[V, Long]]] = Monoid.mapMonoid[String, Map[V, Long]](mapDLMonoid)
}
