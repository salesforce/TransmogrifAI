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
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.{Estimator, Transformer}
import org.apache.spark.ml.linalg.Vectors
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.joda.time.{DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DateMapToUnitCircleVectorizerTest extends OpEstimatorSpec[OPVector, SequenceModel[DateMap, OPVector],
  DateMapToUnitCircleVectorizer[DateMap]] with AttributeAsserts {

  val eps = 1E-4
  val sampleDateTimes = Seq[JDateTime](
    new JDateTime(2018, 2, 11, 0, 0, 0, 0),
    new JDateTime(2018, 11, 28, 6, 0, 0, 0),
    new JDateTime(2018, 2, 17, 12, 0, 0, 0),
    new JDateTime(2017, 4, 17, 18, 0, 0, 0),
    new JDateTime(1918, 2, 13, 3, 0, 0, 0)
  )

  val (inputData, f1) = TestFeatureBuilder(
    sampleDateTimes.map(x => Map("a" -> x.getMillis, "b" -> x.getMillis).toDateMap)
  )

  /**
   * Estimator instance to be tested
   */
  override val estimator = new DateMapToUnitCircleVectorizer[DateMap]().setInput(f1)
    .setTimePeriod(TimePeriod.HourOfDay)
  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  override val expectedResult: Seq[OPVector] = sampleDateTimes
    .map{ v =>
      val rad = DateToUnitCircle.convertToRandians(Option(v.getMillis), TimePeriod.HourOfDay)
      (rad ++ rad).toOPVector
    }

  it should "work with its shortcut as a DateMap" in {
    val output = f1.toUnitCircle(TimePeriod.HourOfDay)
    val transformed = output.originStage.asInstanceOf[DateMapToUnitCircleVectorizer[DateMap]]
      .fit(inputData).transform(inputData)
    val field = transformed.schema(output.name)
    val actual = transformed.collect(output)
    assertNominal(field, Array.fill(actual.head.value.size)(false), actual)
    all (actual.zip(expectedResult).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "work with its shortcut as a DateTimeMap" in {
    val (inputDataDT, f1DT) = TestFeatureBuilder(
      sampleDateTimes.map(x => Map("a" -> x.getMillis, "b" -> x.getMillis).toDateTimeMap)
    )
    val output = f1DT.toUnitCircle(TimePeriod.HourOfDay)
    val transformed = output.originStage.asInstanceOf[DateMapToUnitCircleVectorizer[DateMap]]
      .fit(inputData).transform(inputData)
    val field = transformed.schema(output.name)
    val actual = transformed.collect(output)
    assertNominal(field, Array.fill(actual.head.value.size)(false), actual)
    all (actual.zip(expectedResult).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "make the correct metadata" in {
    val fitted = estimator.fit(inputData)
    val meta = OpVectorMetadata(fitted.getOutputFeatureName, fitted.getMetadata())
    meta.columns.length shouldBe 4
    meta.columns.flatMap(_.grouping) shouldEqual Seq("a", "a", "b", "b")
    meta.columns.flatMap(_.descriptorValue) shouldEqual Seq("x_HourOfDay", "y_HourOfDay", "x_HourOfDay", "y_HourOfDay")
  }

}
