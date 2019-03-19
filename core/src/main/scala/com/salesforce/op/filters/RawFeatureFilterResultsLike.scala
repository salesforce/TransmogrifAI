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

import com.salesforce.op.stages.impl.preparators.CorrelationType
import enumeratum._


/**
 * Keeps the raw feature filter results
 */
trait RawFeatureFilterResultsLike {
  val rawFeatureFilterConfig: RawFeatureFilterConfig
  val rawFeatureDistributions: Seq[FeatureDistribution]
  val rawFeatureFilterMetrics: Seq[RawFeatureFilterMetrics]
  val exclusionReasons: Seq[ExclusionReasons]
}

/**
 * Keeps the configuration for raw feature filter
 */
trait RawFeatureFilterConfigLike {
  val minFill: Double
  val maxFillDifference: Double
  val maxFillRatioDiff: Double
  val maxJSDivergence: Double
  val maxCorrelation: Double
  val correlationType: CorrelationType
  val jsDivergenceProtectedFeatures: Set[String]
  val protectedFeatures: Set[String]
}

/**
 * Keeps the metrics computed by raw feature filter
 */
trait RawFeatureFilterMetricsLike {
  val name: String
  val trainingFillRate: Double
  val trainingNullLabelAbsoluteCorr: Option[Double]
  val scoringFillRate: Option[Double]
  val jsDivergence: Option[Double]
  val fillRateDiff: Option[Double]
  val fillRatioDiff: Option[Double]
}

/**
 * Keeps the reasons why feature was excluded (or not) by raw feature filter
 */
trait ExclusionReasonsLike {
  val name: String
  val trainingUnfilledState: Boolean
  val trainingNullLabelLeaker: Boolean
  val scoringUnfilledState: Boolean
  val jsDivergenceMismatch: Boolean
  val fillRateDiffMismatch: Boolean
  val fillRatioDiffMismatch: Boolean
  val excluded: Boolean
}

sealed trait RawFeatureFilterResultsType extends EnumEntry with Serializable
sealed trait RawFeatureFilterMetricsType extends EnumEntry with Serializable
sealed trait RawFeatureFilterConfigType extends EnumEntry with Serializable
sealed trait ExclusionReasonsType extends EnumEntry with Serializable

object RawFeatureFilterResultsType extends Enum[RawFeatureFilterResultsType] {
  val values: Seq[RawFeatureFilterResultsType] = findValues
}
