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
import com.salesforce.op.features.{FeatureBuilder, FeatureSparkTypes}
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.testkit.{RandomMap, RandomReal, RandomVector}
import com.twitter.algebird.Operators._
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class UniqueCountTest extends FlatSpec with TestSparkContext with UniqueCountFun {
  import spark.implicits._

  implicit val kryo = new KryoSerializer(spark.sparkContext.getConf)

  Spec[UniqueCountFun] should "count uniques" in {
    val data = createData(1000, 10)
    val m = data.first.size
    val (uniqueCounts, total) = countUniques[Double](data, size = m, bits = 12)
    total shouldBe data.count()
    val expected = countUniquesManually(data)
    uniqueCounts.map(_.estimatedSize.toInt) should contain theSameElementsAs expected
  }

  it should "count uniques in maps" in {
    val data = createMapData(1000, 10)
    val m = data.first.size
    val (uniqueCounts, total) = countMapUniques[Double](data, size = m, bits = 12)
    total shouldBe data.count()
    val expected = countUniquesMapManually(data)
    uniqueCounts.flatMap(_.map { case (_, v) => v.estimatedSize.toInt }) should contain theSameElementsAs expected
  }


  private def createData(nRows: Int, nCols: Int): Dataset[Seq[Double]] = {
    val fv = RandomVector.dense(RandomReal.poisson(10), nCols).limit(nRows)
    val seq = fv.map(_.value.toArray.toSeq.map(_.toReal))
    val features = (0 until nCols).map(i => FeatureBuilder.fromRow[Real](i.toString).asPredictor)
    val schema = FeatureSparkTypes.toStructType(features: _ *)
    implicit val rowEncoder = RowEncoder(schema)

    seq.map(p => Row.fromSeq(p.map(FeatureTypeSparkConverter.toSpark))).toDF()
      .map(_.toSeq.map(_.asInstanceOf[Double]))
  }

  private def createMapData(nRows: Int, nCols: Int): Dataset[Seq[Map[String, Double]]] = {
    val poisson = RandomReal.poisson[Real](10)
    val seq = (0 until nCols).map(j =>
      RandomMap.ofReals[Real, RealMap](poisson, 1, 4).withKeys(i => s"$j$i").limit(nRows)
    ).transpose
    val features = (0 until nCols).map(i => FeatureBuilder.fromRow[RealMap](i.toString).asPredictor)
    val schema = FeatureSparkTypes.toStructType(features: _ *)
    implicit val rowEncoder = RowEncoder(schema)

    seq.map(p => Row.fromSeq(p.map(_.value))).toDF()
      .map(_.toSeq.map(_.asInstanceOf[Map[String, Double]]))
  }

  private def countUniquesManually(data: Dataset[Seq[Double]]): Seq[Int] = {
    data.rdd
      .map(_.map(v => Map(v -> 1L)))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => m1 + m2 })
      .map(_.size)
  }

  private def countUniquesMapManually(data: Dataset[Seq[Map[String, Double]]]): Seq[Int] = {
    data.rdd
      .map(_.map(_.flatMap { case (k, v) => Map(k -> Map(v -> 1L)) }))
      .reduce((a, b) => a.zip(b).map { case (m1, m2) => m1 + m2 })
      .flatMap(_.map(_._2.size))
  }
}
