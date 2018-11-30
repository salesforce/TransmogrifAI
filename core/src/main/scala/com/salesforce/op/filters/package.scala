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

package com.salesforce.op

import com.salesforce.op.features.FeatureDistributionType
import com.salesforce.op.utils.json.JsonLike
import com.twitter.algebird.Tuple4Semigroup

import scala.collection.GenTraversable

package object filters {

  private val Scoring: String = s"${FeatureDistributionType.Scoring}"
  private val Training: String = s"${FeatureDistributionType.Training}"

  private[filters] type AllFeatures =
    (Map[FeatureKey, Seq[Double]], Map[FeatureKey, Seq[Double]], Map[FeatureKey, Seq[String]])
  private[filters] type AllSummaries = (
    Double,
    Map[FeatureKey, HistogramSummary],
    Map[FeatureKey, HistogramSummary],
    Map[FeatureKey, TextSummary]
  )
  private[filters] type FeatureKey = (String, Option[String])
  private[filters] type InterpretedReasons = (String, List[String])
  private[filters] type ProcessedSeq = Either[Seq[String], Seq[Double]]

  implicit val allSummariesSg = new Tuple4Semigroup[Double,
    Map[FeatureKey, HistogramSummary],
    Map[FeatureKey, HistogramSummary],
    Map[FeatureKey, TextSummary]]

  case class AllDistributions(
      responseDistributions: Map[FeatureKey, FeatureDistribution],
      numericDistributions: Map[FeatureKey, FeatureDistribution],
      textDistributions: Map[FeatureKey, FeatureDistribution]) {
    def predictorDistributions: Map[FeatureKey, FeatureDistribution] =
      numericDistributions ++ textDistributions

    def predictorKeySet: Set[FeatureKey] = predictorDistributions.keySet
  }

  case class AllReasons(
      featureKey: FeatureKey,
      trainingDistributions: Array[Double],
      trainingSummaryInfo: Array[Double],
      scoringDistributions: Array[Double],
      scoringSummaryInfo: Array[Double],
      trainingFillRate: Double,
      trainingNullLabelCorrelations: Map[FeatureKey, Double],
      scoringFillRate: Option[Double],
      fillRateDifference: Option[Double],
      fillRateRatio: Option[Double],
      jsDistance: Option[Double]) {

    assertDistributions(Training, trainingDistributions, trainingSummaryInfo)
    assertDistributions(Scoring, scoringDistributions, scoringSummaryInfo)

    final def interpret(
      minFill: Double,
      maxCorrelation: Double,
      maxFillDifference: Double,
      maxFillRatioDiff: Double,
      maxJSDistance: Double,
      jsDistanceProtectedFeatures: Set[String]): InterpretedReasons = {
      val description =
        interpretDistributions(Training, trainingDistributions, trainingSummaryInfo) +
          interpretDistributions(Scoring, scoringDistributions, scoringSummaryInfo) +
          Seq(s"Feature Key = $featureKey",
            s"$Training Fill Rate = $trainingFillRate",
            s"$Training Null Label Correlations = $trainingNullLabelCorrelations",
            s"$Scoring Fill Rate = $scoringFillRate",
            s"Fill Rate Difference = $fillRateDifference",
            s"Fill Rate Ratio = $fillRateRatio",
            s"JS Distance = $jsDistance").mkString("\n")
      val interpretedReasons = List(
        interpretFillRate(Training, minFill, Option(trainingFillRate)),
        interpretNullLabelCorrelations(maxCorrelation),
        interpretFillRate(Scoring, minFill, scoringFillRate),
        interpretJSDistance(jsDistance, jsDistanceProtectedFeatures, maxJSDistance),
        interpretFillRateDifference(fillRateDifference, maxFillDifference),
        interpretFillRateRatio(fillRateRatio, maxFillRatioDiff)
      ).flatten

      description -> interpretedReasons
    }

    private def assertDistributions(
      distributionType: String,
      distributions: Array[Double],
      summaryInfo: Array[Double]): Unit = assert(
        distributions.length == summaryInfo.length,
        s"$distributionType distributions should be in bijection with summary info")

    private def interpretDistributions(
      distributionType: String,
      distributions: Array[Double],
      summaryInfo: Array[Double]): String =
      s"$distributionType Data: ${summaryInfo.zip(distributions).toList}\n"

    private def interpretFillRate(
      distributionType: String,
      minFill: Double,
      fillRateOpt: Option[Double]): GenTraversable[String] =
      fillRateOpt.flatMap(handleNaN(_).flatMap { fillRate =>
        if (fillRate < minFill) {
          Option(s"$distributionType fill rate did not meet minimum required: $minFill")
        } else {
          None
        }
      })

    private def interpretFillRateDifference(
      fillRateDifferenceOpt: Option[Double],
      maxFillDifference: Double): GenTraversable[String] =
      fillRateDifferenceOpt.flatMap(handleNaN(_).flatMap { fillRateDifference =>
        if (fillRateDifference > maxFillDifference) {
          Option(s"fill rate difference exceeded max allowed ($maxFillDifference)")
        } else {
          None
        }
      })

    private def interpretFillRateRatio(
      fillRateRatioOpt: Option[Double],
      maxFillRatioDiff: Double): GenTraversable[String] =
      fillRateRatioOpt.flatMap(handleNaN(_).flatMap { fillRateRatio =>
        if (fillRateRatio > maxFillRatioDiff) {
          Option(s"fill ratio difference exceeded max allowed ($maxFillRatioDiff)")
        } else {
          None
        }

      })

    private def interpretJSDistance(
      jsdOpt: Option[Double],
      jsDistanceProtectedFeatures: Set[String],
      maxJSDistance: Double): GenTraversable[String] =
      jsdOpt.flatMap(handleNaN(_).flatMap { jsd =>
        if (!jsDistanceProtectedFeatures.contains(featureKey._1) && jsd > maxJSDistance) {
          Option(s"JS Distance between $Training and $Scoring exceeded max allowed ($maxJSDistance)")
        } else {
          None
        }
      })

    private def interpretNullLabelCorrelations(maxCorrelation: Double): GenTraversable[String] =
      trainingNullLabelCorrelations.flatMap {
        case (featureKey, nullLabelCorrelation) => handleNaN(nullLabelCorrelation).flatMap { corr =>
          if (corr > maxCorrelation) {
            Option(
              s"$Training null label correlation for response $featureKey, $corr exceeded maximum: $maxCorrelation")
          } else {
            None
          }
        }
      }

    private def handleNaN(input: Double): Option[Double] = if (input.isNaN) None else Option(input)
  }
}
