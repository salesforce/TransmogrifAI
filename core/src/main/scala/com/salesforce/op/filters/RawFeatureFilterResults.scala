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

/**
 * Contains configuration and results from RawFeatureFilter
 *
 * @param rawFeatureFilterConfig  configuration settings for RawFeatureFilter
 * @param featureDistributions    feature distributions calculated from training data
 * @param exclusionReasons        results of RawFeatureFilter tests (reasons why feature is dropped or not)
 */
case class RawFeatureFilterResults
(
  rawFeatureFilterConfig: RawFeatureFilterConfig,
  featureDistributions: Seq[FeatureDistribution],
  exclusionReasons: Seq[ExclusionReasons]
)

/**
 * Contains configuration settings for RawFeatureFilter
 */
case class RawFeatureFilterConfig
(
  minFill: Double,
  maxFillDifference: Double,
  maxFillRatioDiff: Double,
  maxJSDivergence: Double,
  maxCorrelation: Double,
  correlationType: CorrelationType,
  jsDivergenceProtectedFeatures: Set[String],
  protectedFeatures: Set[String]
)

/**
 * Contains results of Raw Feature Filter tests for a given feature
 *
 * @param trainingUnfilled              training fill rate did not meet min required
 * @param scoringUnfilled               scoring fill rate did not meet min required
 * @param distribMismatchJSDivergence   distribution mismatch: JS Divergence exceeded max allowed
 * @param distribMismatchFillRateDiff   distribution mismatch: fill rate difference exceeded max allowed
 * @param distribMismatchFillRatioDiff  distribution mismatch: fill ratio difference exceeded max allowed
 * @param nullLabelCorrelation          null indicator correlation (absolute) exceeded max allowed
 * @param excluded                      feature excluded after failing one or more tests
 */
case class ExclusionReasons
(
  trainingUnfilled: Boolean = false,
  scoringUnfilled: Boolean = false,
  distribMismatchJSDivergence: Boolean = false,
  distribMismatchFillRateDiff: Boolean = false,
  distribMismatchFillRatioDiff: Boolean = false,
  nullLabelCorrelation: Boolean = false,
  excluded: Boolean = false
)