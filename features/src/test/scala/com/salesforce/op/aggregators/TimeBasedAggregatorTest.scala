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

package com.salesforce.op.aggregators

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.stages.FeatureGeneratorStage
import com.salesforce.op.test.TestCommon
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TimeBasedAggregatorTest extends FlatSpec with TestCommon {

  private val data = Seq(TimeBasedTest(100L, 1.0, "a", Map("a" -> "a")),
    TimeBasedTest(200L, 2.0, "b", Map("b" -> "b")),
    TimeBasedTest(300L, 3.0, "c", Map("c" -> "c")),
    TimeBasedTest(400L, 4.0, "d", Map("d" -> "d")),
    TimeBasedTest(500L, 5.0, "e", Map("e" -> "e")),
    TimeBasedTest(600L, 6.0, "f", Map("f" -> "f"))
  )

  private val timeExt = Option((d: TimeBasedTest) => d.time)

  Spec[LastAggregator[_]] should "return the most recent event" in {
    val feature = FeatureBuilder.Real[TimeBasedTest].extract(_.real.toRealNN)
      .aggregate(LastReal).asPredictor
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(data, timeExt, CutOffTime.NoCutoff())
    extracted shouldBe Real(Some(6.0))
  }

  it should "return the most recent event within the time window" in {
    val feature = FeatureBuilder.Text[TimeBasedTest].extract(_.string.toText)
      .aggregate(LastText).asResponse
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(data, timeExt, CutOffTime.UnixEpoch(300L),
      responseWindow = Option(new Duration(201L)))
    extracted shouldBe Text(Some("e"))
  }

  it should "return the feature type empty value when no events are passed in" in {
    val feature = FeatureBuilder.TextMap[TimeBasedTest].extract(_.map.toTextMap)
      .aggregate(LastTextMap).asPredictor
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(Seq(), timeExt, CutOffTime.NoCutoff())
    extracted shouldBe TextMap.empty
  }

  Spec[FirstAggregator[_]] should "return the first event" in {
    val feature = FeatureBuilder.TextAreaMap[TimeBasedTest].extract(_.map.toTextAreaMap)
      .aggregate(FirstTextAreaMap).asResponse
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(data, timeExt, CutOffTime.UnixEpoch(301L))
    extracted shouldBe TextAreaMap(Map("d" -> "d"))
  }

  it should "return the first event within the time window" in {
    val feature = FeatureBuilder.Currency[TimeBasedTest].extract(_.real.toCurrency)
      .aggregate(FirstCurrency).asPredictor
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(data, timeExt, CutOffTime.UnixEpoch(400L),
      predictorWindow = Option(new Duration(201L)))
    extracted shouldBe Currency(Some(2.0))
  }

  it should "return the feature type empty value when no events are passed in" in {
    val feature = FeatureBuilder.State[TimeBasedTest].extract(_.string.toState)
      .aggregate(FirstState).asPredictor
    val aggregator = feature.originStage.asInstanceOf[FeatureGeneratorStage[TimeBasedTest, _]].featureAggregator
    val extracted = aggregator.extract(Seq(), timeExt, CutOffTime.NoCutoff())
    extracted shouldBe State.empty
  }
}

case class TimeBasedTest(time: Long, real: Double, string: String, map: Map[String, String])


