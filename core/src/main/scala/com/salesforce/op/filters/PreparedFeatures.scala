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


import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureDistributionType, TransientFeature}
import com.salesforce.op.stages.impl.feature.{DateToUnitCircle, TextTokenizer, TimePeriod}
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.utils.text.Language
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.Row

/**
 * Class representing processed responses and predictors keyed by their respective feature key
 *
 * @param responses prepared responses
 * @param predictors prepared predictors
 */
private[filters] case class PreparedFeatures
(
  responses: Map[FeatureKey, ProcessedSeq],
  predictors: Map[FeatureKey, ProcessedSeq]
) {

  /**
   * Computes summaries keyed by feature keys for this observation.
   *
   * @return pair consisting of response and predictor summaries (in this order)
   */
  def summaries: (Map[FeatureKey, Summary], Map[FeatureKey, Summary]) =
    responses.map { case (k, s) => k -> Summary(s) } -> predictors.map { case (k, s) => k -> Summary(s) }

  /**
   * Computes vector of size responseKeys.length + predictorKeys.length. The first responses.length
   * values are the actual response values (nulls replaced with 0.0). Its (i + responses.length)th value
   * is 1 iff. the predictor associated to ith f eature key is null, for i >= 0.
   *
   * @param responseKeys  response feature keys
   * @param predictorKeys set of all predictor keys needed for constructing binary vector
   * @return null label-leakage correlation vector
   */
  def getNullLabelLeakageVector(responseKeys: Array[FeatureKey], predictorKeys: Array[FeatureKey]): Vector = {
    val responseValues = responseKeys.map(responses.get(_).collect {
      case Right(Seq(d)) => d
    }.getOrElse(0.0))
    val predictorNullIndicatorValues = predictorKeys.map(predictors.get(_).map(_ => 0.0).getOrElse(1.0))

    Vectors.dense(responseValues ++ predictorNullIndicatorValues)
  }

  /**
   * Generates a pair of feature distribution arrays. The first element is associated to responses,
   * and the second to predictors.
   *
   * @param responseSummaries       global feature metadata
   * @param predictorSummaries      set of feature summary statistics (derived from metadata)
   * @param bins                    number of bins to put numerics into
   * @param textBinsFormula         formula to compute the text features bin size.
   *                                Input arguments are [[Summary]] and number of bins to use in
   *                                computing feature distributions (histograms for numerics, hashes for strings).
   *                                Output is the bins for the text features.
   * @param featureDistributionType feature distribution type: training or scoring
   * @return a pair consisting of response and predictor feature distributions (in this order)
   */
  def getFeatureDistributions(
    responseSummaries: Array[(FeatureKey, Summary)],
    predictorSummaries: Array[(FeatureKey, Summary)],
    bins: Int,
    textBinsFormula: (Summary, Int) => Int,
    featureDistributionType: FeatureDistributionType
  ): (Array[FeatureDistribution], Array[FeatureDistribution]) = {

    def featureDistributions(
      features: Map[FeatureKey, ProcessedSeq],
      summaries: Array[(FeatureKey, Summary)]
    ): Array[FeatureDistribution] = summaries.map { case (featureKey, summary) =>
      FeatureDistribution.fromSummary(
        summary = summary,
        featureKey = featureKey,
        value = features.get(featureKey),
        bins = bins,
        textBinsFormula = textBinsFormula,
        `type` = featureDistributionType
      )
    }

    val responseFeatureDistributions = featureDistributions(responses, responseSummaries)
    val predictorFeatureDistributions = featureDistributions(predictors, predictorSummaries)
    responseFeatureDistributions -> predictorFeatureDistributions
  }
}

private[filters] object PreparedFeatures {

  /**
   * Retrieve prepared features from a given data frame row and transient features partition
   * into responses and predictors.
   *
   * @param row data frame row
   * @param responses transient features derived from responses
   * @param predictors transient features derived from predictors
   * @param timePeriod optional time period to use raw feature binning, otherwise standard numeric transformation
   *                   is applied
   * @return set of prepared features
   */
  def apply(
    row: Row,
    responses: Array[TransientFeature],
    predictors: Array[TransientFeature],
    timePeriod: Option[TimePeriod]
  ): PreparedFeatures = {
    val empty: Map[FeatureKey, ProcessedSeq] = Map.empty
    val preparedResponses = responses.foldLeft(empty) { case (map, feature) =>
      val converter = FeatureTypeSparkConverter.fromFeatureTypeName(feature.typeName)
      map ++ prepareFeature(feature.name, row.getFeatureType(feature)(converter), timePeriod)
    }
    val preparedPredictors = predictors.foldLeft(empty) { case (map, feature) =>
      val converter = FeatureTypeSparkConverter.fromFeatureTypeName(feature.typeName)
      map ++ prepareFeature(feature.name, row.getFeatureType(feature)(converter), timePeriod)
    }

    PreparedFeatures(responses = preparedResponses, predictors = preparedPredictors)
  }

  /**
   * Turn features into a sequence that will have stats computed on it based on the type of the feature
   *
   * @param name feature name
   * @param value feature value
   * @param timePeriod optional time period to use raw feature binning, otherwise standard numeric transformation
   *                   is applied
   * @tparam T type of the feature
   * @return tuple containing whether the feature was empty and a sequence of either doubles or strings
   */
  private def prepareFeature[T <: FeatureType](
    name: String,
    value: T,
    timePeriod: Option[TimePeriod]): Map[FeatureKey, ProcessedSeq] =
    value match {
      case SomeValue(v: DenseVector) => Map((name, None) -> Right(v.toArray.toSeq))
      case SomeValue(v: SparseVector) => Map((name, None) -> Right(v.indices.map(_.toDouble).toSeq))
      case ft@SomeValue(_) => ft match {
        case v: Text => Map((name, None) -> Left(v.value.toSeq.flatMap(tokenize)))
        case v: Date => Map((name, None) -> Right(v.value.map(prepareDateValue(_, timePeriod)).toSeq))
        case v: OPNumeric[_] => Map((name, None) -> Right(v.toDouble.toSeq))
        case v: Geolocation => Map((name, None) -> Right(v.value))
        case v: TextList => Map((name, None) -> Left(v.value))
        case v: DateList => Map((name, None) -> Right(v.value.map(prepareDateValue(_, timePeriod))))
        case v: MultiPickList => Map((name, None) -> Left(v.value.toSeq))
        case v: MultiPickListMap => v.value.map { case (k, e) => (name, Option(k)) -> Left(e.toSeq) }
        case v: GeolocationMap => v.value.map{ case (k, e) => (name, Option(k)) -> Right(e) }
        case v: DateMap =>
          v.value.map { case (k, e) => (name, Option(k)) -> Right(Seq(prepareDateValue(e, timePeriod))) }
        case v: OPMap[_] => v.value.map { case (k, e) => e match {
          case d: Double => (name, Option(k)) -> Right(Seq(d))
          // Do not need to distinguish between string map types, all text is tokenized for distribution calculation
          case s: String => (name, Option(k)) -> Left(tokenize(s))
          case l: Long => (name, Option(k)) -> Right(Seq(l.toDouble))
          case b: Boolean => (name, Option(k)) -> Right(Seq(if (b) 1.0 else 0.0))
        }}
        case _ => throw new RuntimeException(s"Feature type $value is not supported in RawFeatureFilter")
      }
      case _ => Map.empty
    }

  /**
   * Tokenizes an input string.
   *
   * @param s input string
   * @return array of string tokens
   */
  private def tokenize(s: String) = TextTokenizer.Analyzer.analyze(s, Language.Unknown)

  private def prepareDateValue(timestamp: Long, timePeriod: Option[TimePeriod]): Double =
    timePeriod match {
      case Some(period) => DateToUnitCircle.convertToBin(timestamp, period)
      case None => timestamp.toDouble
    }
}
