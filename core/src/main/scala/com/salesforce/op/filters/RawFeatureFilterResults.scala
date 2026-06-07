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

package com.salesforce.op.filters

import com.salesforce.op.features.FeatureDistributionType
import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.utils.json.{EnumEntrySerializer, SpecialDoubleSerializer}
import com.twitter.algebird.MomentsSerializer
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.{DefaultFormats, Formats}

import scala.util.Try

/**
 * Contains configuration and results from RawFeatureFilter
 *
 * @param rawFeatureFilterConfig  configuration settings for RawFeatureFilter
 * @param rawFeatureDistributions feature distributions calculated from training data
 * @param rawFeatureFilterMetrics feature metrics calculated by RawFeatureFilter
 * @param exclusionReasons        results of RawFeatureFilter tests (reasons why feature is dropped or not)
 */
case class RawFeatureFilterResults
(
  rawFeatureFilterConfig: RawFeatureFilterConfig = RawFeatureFilterConfig(),
  rawFeatureDistributions: Seq[FeatureDistribution] = Seq.empty,
  rawFeatureFilterMetrics: Seq[RawFeatureFilterMetrics] = Seq.empty,
  exclusionReasons: Seq[ExclusionReasons] = Seq.empty
)

trait RawFeatureFilterFormats {
  implicit val jsonFormats: Formats = DefaultFormats +
    new SpecialDoubleSerializer +
    EnumEntrySerializer.json4s[CorrelationType](CorrelationType) +
    FeatureDistribution.fieldSerializer ++
    FeatureDistribution.serializers

}

object RawFeatureFilterResults extends RawFeatureFilterFormats {

  /**
   * RawFeatureFilterResults to json
   *
   * @param results raw feature filter results
   * @return json array
   */
  def toJson(results: RawFeatureFilterResults): String = Serialization.write[RawFeatureFilterResults](results)

  /**
   * RawFeatureFilterResults from json
   *
   * @param json json
   * @return raw feature filter results
   */
  def fromJson(json: String): Try[RawFeatureFilterResults] = Try { Serialization.read[RawFeatureFilterResults](json) }

}

/**
 * Contains configuration settings for Raw Feature Filter
 */
case class RawFeatureFilterConfig
(
  minFill: Double = 0.0,
  maxFillDifference: Double = Double.PositiveInfinity,
  maxFillRatioDiff: Double = Double.PositiveInfinity,
  maxJSDivergence: Double = 1.0,
  maxCorrelation: Double = 1.0,
  correlationType: CorrelationType = CorrelationType.Pearson,
  jsDivergenceProtectedFeatures: Seq[String] = Seq.empty,
  protectedFeatures: Seq[String] = Seq.empty
)

object RawFeatureFilterConfig extends RawFeatureFilterFormats {

  /**
   * Converts case class constructor to a Map. Values are converted to String
   *
   * @return Map[String, String]
   */
  def toStringMap(config: RawFeatureFilterConfig): Map[String, String] = {
    val params = parse(Serialization.write[RawFeatureFilterConfig](config)).extract[Map[String, Any]]
    params.map { case (key, value) => (key, value.toString) }
  }

  /**
   * Summarize RawFeatureFilterConfig in format of stageInfo; this info will be passed alongside stage info in
   * ModelInsights
   *
   * @return Map[String, Map[String, Any] ]
   */
  def toStageInfo(config: RawFeatureFilterConfig): Map[String, Map[String, Any]] = {
    Map(RawFeatureFilter.stageName -> Map("uid" -> RawFeatureFilter.uid, "params" -> toStringMap(config)))
  }

}

/**
 * Contains raw feature metrics computing in Raw Feature Filter
 *
 * @param name                          feature name
 * @param key                           map key associated with distribution (when the feature is a map)
 * @param trainingFillRate              proportion of values that are null in the training distribution
 * @param trainingNullLabelAbsoluteCorr correlation between null indicator and the label in the training distribution
 * @param scoringFillRate               proportion of values that are null in the scoring distribution
 * @param jsDivergence                  Jensen-Shannon (JS) divergence between the training and scoring distributions
 * @param fillRateDiff                  absolute difference in fill rates between the training and scoring distributions
 * @param fillRatioDiff                 ratio of difference in fill rates between the training and scoring distributions
 */
case class RawFeatureFilterMetrics
(
  name: String,
  key: Option[String],
  trainingFillRate: Double,
  trainingNullLabelAbsoluteCorr: Option[Double],
  scoringFillRate: Option[Double],
  jsDivergence: Option[Double],
  fillRateDiff: Option[Double],
  fillRatioDiff: Option[Double]
)

/**
 * Contains results of Raw Feature Filter tests for a given feature
 *
 * @param name                    feature name
 * @param key                     map key associated with distribution (when the feature is a map)
 * @param trainingUnfilledState   training fill rate did not meet min required
 * @param trainingNullLabelLeaker null indicator correlation (absolute) exceeded max allowed
 * @param scoringUnfilledState    scoring fill rate did not meet min required
 * @param jsDivergenceMismatch    distribution mismatch: JS Divergence exceeded max allowed
 * @param fillRateDiffMismatch    distribution mismatch: fill rate difference exceeded max allowed
 * @param fillRatioDiffMismatch   distribution mismatch: fill ratio difference exceeded max allowed
 * @param excluded                feature excluded after failing one or more tests
 */
case class ExclusionReasons
(
  name: String,
  key: Option[String],
  trainingUnfilledState: Boolean,
  trainingNullLabelLeaker: Boolean,
  scoringUnfilledState: Boolean,
  jsDivergenceMismatch: Boolean,
  fillRateDiffMismatch: Boolean,
  fillRatioDiffMismatch: Boolean,
  excluded: Boolean
)
