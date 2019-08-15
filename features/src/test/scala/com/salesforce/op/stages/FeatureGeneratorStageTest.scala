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

package com.salesforce.op.stages

import com.salesforce.op.aggregators.{CutOffTime, Event, MonoidAggregatorDefaults}
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, JValue}
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.json4s.JValue
import org.json4s.JsonAST.{JInt, JObject}
import org.json4s.JsonDSL._
import org.junit.runner.RunWith

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Try


@RunWith(classOf[JUnitRunner])
class FeatureGeneratorStageTest extends FlatSpec with TestSparkContext {

  val (ds, features) = TestFeatureBuilder.random()()

  type FeaturesAndGenerators = Array[(Feature[_ <: FeatureType], FeatureGeneratorStage[Row, _ <: FeatureType])]

  val featuresAndGenerators: FeaturesAndGenerators =
    features
      .map(f => f -> f.originStage)
      .collect { case (f, fg: FeatureGeneratorStage[Row, _]@unchecked) => f -> fg }

  val rows = ds.collect()

  Spec[FeatureGeneratorStage[_, _]] should "be the origin stage for raw features" in {
    features.length shouldBe featuresAndGenerators.length
  }

  it should "extract feature values" in assertExtractFeatures(featuresAndGenerators)

  it should "aggregate feature values" in assertAggregateFeatures(featuresAndGenerators)

  it should "serialize to/from json, then extract and aggregate feature values" in {
    val recovered: FeaturesAndGenerators =
      for {(feature, featureGenerator) <- featuresAndGenerators} yield {
        val featureGenJson = featureGenerator.write.asInstanceOf[OpPipelineStageWriter].writeToJsonString("")
        val recoveredStage = new OpPipelineStageReader(Seq.empty).loadFromJsonString(featureGenJson, "")
        recoveredStage shouldBe a[FeatureGeneratorStage[_, _]]
        feature -> recoveredStage.asInstanceOf[FeatureGeneratorStage[Row, _ <: FeatureType]]
      }
    assertExtractFeatures(recovered)
    assertAggregateFeatures(recovered)
  }

  it should "serialize to/from json with a parametrized extract function" in {
    val multiplier = 10
    val multiplied = FeatureBuilder.Integral[Int].extract(new IntMultiplyExtractor(multiplier)).asPredictor
    val featureGenerator = multiplied.originStage
    val featureGenJson = featureGenerator.write.asInstanceOf[OpPipelineStageWriter].writeToJsonString("")
    val recoveredStage = new OpPipelineStageReader(Seq.empty).loadFromJsonString(featureGenJson, "")
    recoveredStage shouldBe a[FeatureGeneratorStage[_, _]]
    val extractFn = recoveredStage.asInstanceOf[FeatureGeneratorStage[Int, Integral]].extractFn
    extractFn shouldBe a[IntMultiplyExtractor]
    extractFn.apply(7) shouldBe 70.toIntegral
  }

  def assertExtractFeatures(fgs: FeaturesAndGenerators): Unit = {
    for {(feature, featureGenerator) <- fgs} {
      rows.map { row =>
        val featureValue = featureGenerator.extractFn(row)
        featureValue shouldBe a[FeatureType]
        row.getAny(feature.name) shouldBe FeatureTypeSparkConverter.toSpark(featureValue)
      }
    }
  }

  def assertAggregateFeatures(fgs: FeaturesAndGenerators): Unit = {
    for {(feature, featureGenerator) <- featuresAndGenerators} {
      val fa = featureGenerator.featureAggregator
      val expectedValue = fa.extract(rows, timeStampFn = None, cutOffTime = CutOffTime.NoCutoff())

      val ftt = feature.wtt.asInstanceOf[WeakTypeTag[FeatureType]]
      val rowVals = rows.map(r => FeatureTypeSparkConverter()(ftt).fromSpark(r.getAny(feature.name)))
      val events = rowVals.map(Event(0L, _))
      val aggr = MonoidAggregatorDefaults.aggregatorOf(ftt)
      val aggregatedValue = aggr(events)

      aggregatedValue shouldEqual expectedValue
    }
  }

}

@ReaderWriter(classOf[IntMultiplyExtractorReadWrite])
class IntMultiplyExtractor(val multiplier: Int) extends Function1[Int, Integral] {
  def apply(i: Int): Integral = (i * multiplier).toIntegral
}

class IntMultiplyExtractorReadWrite extends ValueReaderWriter[IntMultiplyExtractor] {
  implicit val formats = DefaultFormats
  def read(valueClass: Class[IntMultiplyExtractor], json: JValue): Try[IntMultiplyExtractor] = Try {
    new IntMultiplyExtractor((json \ "multiplier").extract[Int])
  }
  def write(value: IntMultiplyExtractor): Try[JValue] = Try {
    "multiplier" -> JInt(value.multiplier)
  }
}
