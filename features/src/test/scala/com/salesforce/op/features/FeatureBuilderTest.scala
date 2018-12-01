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

package com.salesforce.op.features

import java.util

import com.salesforce.op.aggregators._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.FeatureGeneratorStage
import com.salesforce.op.test.{Passenger, TestSparkContext}
import com.twitter.algebird.MonoidAggregator
import org.apache.spark.sql.{DataFrame, Row}
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class FeatureBuilderTest extends FlatSpec with TestSparkContext {
  private val name = "feature"
  private val passenger =
    Passenger.newBuilder()
      .setPassengerId(0).setGender("Male").setAge(1).setBoarded(2).setHeight(3)
      .setWeight(4).setDescription("").setSurvived(1).setRecordDate(4)
      .setStringMap(new util.HashMap[String, String]())
      .setNumericMap(new util.HashMap[String, java.lang.Double]())
      .setBooleanMap(new util.HashMap[String, java.lang.Boolean]())
      .build()

  import spark.implicits._
  private val data: DataFrame = Seq(FeatureBuilderContainerTest("blah1", 10, d = 2.0)).toDS.toDF()

  Spec(FeatureBuilder.getClass) should "build a simple feature with a custom name" in {
    val f1 = FeatureBuilder.Real[Passenger]("a").extract(p => Option(p.getAge).map(_.toDouble).toReal).asPredictor
    assertFeature[Passenger, Real](f1)(in = passenger, out = 1.toReal, name = "a")

    val f2 = FeatureBuilder[Passenger, Real]("b").extract(p => Option(p.getAge).map(_.toDouble).toReal).asResponse
    assertFeature[Passenger, Real](f2)(in = passenger, isResponse = true, out = 1.toReal, name = "b")
  }

  it should "build a simple feature using macro" in {
    val feature = FeatureBuilder.Real[Passenger].extract(p => Option(p.getAge).map(_.toDouble).toReal).asResponse
    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, isResponse = true)
  }

  it should "build a simple feature from Row with a custom name" in {
    val feature = FeatureBuilder.fromRow[Text](name = "feat", Some(1)).asPredictor
    assertFeature[Row, Text](feature)(name = "feat", in = Row(1.0, "2"), out = "2".toText, isResponse = false)
  }

  it should "build a simple feature from Row using macro" in {
    val feature = FeatureBuilder.fromRow[Real](0).asResponse
    assertFeature[Row, Real](feature)(name = name, in = Row(1.0, "2"), out = 1.toReal, isResponse = true)
  }

  it should "build text & numeric features from a dataframe" in {
    val row = data.head()
    val (label, Array(fs, fl)) = FeatureBuilder.fromDataFrame[RealNN](data, response = "d")
    assertFeature(label)(name = "d", in = row, out = 2.0.toRealNN, isResponse = true)
    assertFeature(fs.asInstanceOf[Feature[Text]])(name = "s", in = row, out = "blah1".toText, isResponse = false)
    assertFeature(fl.asInstanceOf[Feature[Integral]])(name = "l", in = row, out = 10.toIntegral, isResponse = false)
  }

  it should "build time & date features from a dataframe " in {
    val ts = java.sql.Timestamp.valueOf("2018-12-02 11:05:33.523")
    val dt = java.sql.Date.valueOf("2018-12-1")
    val df = spark.createDataFrame(Seq((1.0, ts, dt)))
    val Array(lblName, tsName, dtName) = df.schema.fieldNames
    val row = df.head()
    val (label, Array(time, date)) = FeatureBuilder.fromDataFrame[RealNN](df, response = lblName)
    assertFeature(label)(name = lblName, in = row, out = 1.0.toRealNN, isResponse = true)
    assertFeature(time.asInstanceOf[Feature[DateTime]])(
      name = tsName, in = row, out = ts.getTime.toDateTime, isResponse = false)
    assertFeature(date.asInstanceOf[Feature[Date]])(
      name = dtName, in = row, out = dt.getTime.toDate, isResponse = false)
  }

  it should "error on invalid response" in {
    intercept[RuntimeException](FeatureBuilder.fromDataFrame[RealNN](data, response = "non_existent"))
      .getMessage shouldBe "Response feature 'non_existent' was not found in dataframe schema"
    intercept[RuntimeException](FeatureBuilder.fromDataFrame[RealNN](data, response = "s")).getMessage shouldBe
      "Response feature 's' is of type com.salesforce.op.features.types.Text, " +
        "but expected com.salesforce.op.features.types.RealNN"
    intercept[RuntimeException](FeatureBuilder.fromDataFrame[Text](data, response = "d")).getMessage shouldBe
      "Response feature 'd' is of type com.salesforce.op.features.types.RealNN, " +
        "but expected com.salesforce.op.features.types.Text"
  }

  it should "return a default if extract throws an exception" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge / 0).map(_.toDouble).toReal, 123.toReal)
        .asResponse

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 123.toReal, isResponse = true)
  }

  it should "build an aggregated feature" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal).aggregate(MaxReal)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, aggregator = _ => MaxReal)
  }

  it should "build an aggregated feature with an aggregate window" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .window(new Duration(123))
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name,
      in = passenger, out = 1.toReal, aggregateWindow = Some(new Duration(123)))
  }

  it should "build an aggregated feature with a custom aggregator" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .aggregate(MaxReal)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, aggregator = _ => MaxReal)
  }

  it should "build an aggregated feature with a custom aggregate function" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .aggregate((v1, _) => v1)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal,
      aggregator = _ => feature.originStage.asInstanceOf[FeatureGeneratorStage[Passenger, Real]].aggregator
    )
  }

  it should "build an aggregated feature with a custom aggregate function with zero" in {
    val feature = FeatureBuilder.Real[Passenger]
      .extract(p => Option(p.getAge).map(_.toDouble).toReal)
      .aggregate(Real.empty.v, (v1, _) => v1)
      .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal,
      aggregator = _ => feature.originStage.asInstanceOf[FeatureGeneratorStage[Passenger, Real]].aggregator
    )
  }

}

/**
 * Assert feature instance on a given input/output
 */
object assertFeature extends Matchers {

  /**
   * Assert feature instance on a given input/output
   *
   * @param f               feature to assert
   * @param in              input value
   * @param out             expected output value
   * @param name            expected name
   * @param isResponse      is expected to be a response
   * @param aggregator      expected aggregator
   * @param aggregateWindow expected aggregate window
   * @param tti             expected input typetag
   * @param wtt             expected output typetag
   * @tparam I input type
   * @tparam O output feature type
   */
  def apply[I, O <: FeatureType](f: FeatureLike[O])(
    in: I, out: O, name: String, isResponse: Boolean = false,
    aggregator: WeakTypeTag[O] => MonoidAggregator[Event[O], _, O] =
    (wtt: WeakTypeTag[O]) => MonoidAggregatorDefaults.aggregatorOf[O](wtt),
    aggregateWindow: Option[Duration] = None
  )(implicit tti: WeakTypeTag[I], wtt: WeakTypeTag[O]): Unit = {
    f.name shouldBe name
    f.isResponse shouldBe isResponse
    f.parents shouldBe Nil
    f.uid.startsWith(wtt.tpe.dealias.toString.split("\\.").last) shouldBe true
    f.wtt.tpe =:= wtt.tpe shouldBe true
    f.isRaw shouldBe true
    f.typeName shouldBe wtt.tpe.typeSymbol.fullName

    f.originStage shouldBe a[FeatureGeneratorStage[_, _ <: FeatureType]]
    val fg = f.originStage.asInstanceOf[FeatureGeneratorStage[I, O]]
    fg.tti shouldBe tti
    fg.aggregator shouldBe aggregator(wtt)
    fg.extractFn(in) shouldBe out
    fg.extractSource.nonEmpty shouldBe true // TODO we should eval the code here: eval(fg.extractSource)(in)
    fg.getOutputFeatureName shouldBe name
    fg.outputIsResponse shouldBe isResponse
    fg.aggregateWindow shouldBe aggregateWindow
    fg.uid.startsWith(classOf[FeatureGeneratorStage[I, O]].getSimpleName) shouldBe true
  }

}
