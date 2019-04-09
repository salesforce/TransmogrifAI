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

import org.scalactic.Equality
import org.scalatest.{FlatSpec, Matchers}

/**
 * Contains utility functions for comparing two RawFeatureFilterResults
 */
object RawFeatureFilterResultsComparison extends FlatSpec with Matchers {

  class OptionDoubleEquality[T <: Option[Double]] extends Equality[T] {
    def areEqual(a: T, b: Any): Boolean = b match {
      case None => a.isEmpty
      case Some(d: Double) => (a.exists(_.isNaN) && d.isNaN) || a.contains(d)
      case _ => false
    }
  }

  implicit val otherDoubleEquality = new OptionDoubleEquality[Option[Double]]
  implicit val someDoubleEquality = new OptionDoubleEquality[Some[Double]]

  def compareConfig(c1: RawFeatureFilterConfig, c2: RawFeatureFilterConfig): Unit = {
    c1.minFill shouldBe c2.minFill
    c1.maxFillDifference shouldBe c2.maxFillDifference
    c1.maxFillRatioDiff shouldBe c2.maxFillRatioDiff
    c1.maxJSDivergence shouldBe c2.maxJSDivergence
    c1.maxCorrelation shouldBe c2.maxCorrelation
    c1.correlationType shouldBe c2.correlationType
    c1.jsDivergenceProtectedFeatures shouldBe c2.jsDivergenceProtectedFeatures
    c1.protectedFeatures shouldBe c2.protectedFeatures
  }

  def compareDistributions(d1: FeatureDistribution, d2: FeatureDistribution): Unit = {
    d1.name shouldEqual d2.name
    d1.key shouldEqual d2.key
    d1.count shouldEqual d2.count
    d1.nulls shouldEqual d2.nulls
    d1.distribution shouldEqual d2.distribution
    d1.summaryInfo shouldEqual d2.summaryInfo
  }

  def compareSeqDistributions(d1: Seq[FeatureDistribution], d2: Seq[FeatureDistribution]): Unit = {
    d1.zip(d2).foreach { case (a, b) => compareDistributions(a, b) }
  }

  def compareMetrics(m1: RawFeatureFilterMetrics, m2: RawFeatureFilterMetrics): Unit = {
    m1.name shouldBe m2.name
    m1.trainingFillRate shouldBe m2.trainingFillRate
    m1.trainingNullLabelAbsoluteCorr shouldEqual m2.trainingNullLabelAbsoluteCorr
    m1.scoringFillRate shouldEqual m2.scoringFillRate
    m1.jsDivergence shouldEqual m2.jsDivergence
    m1.fillRateDiff shouldEqual m2.fillRateDiff
    m1.fillRatioDiff shouldEqual m2.fillRatioDiff
  }

  def compareSeqMetrics(m1: Seq[RawFeatureFilterMetrics], m2: Seq[RawFeatureFilterMetrics]): Unit = {
    m1.zip(m2).foreach { case (a, b) => compareMetrics(a, b) }
  }

  def compareExclusionReasons(er1: ExclusionReasons, er2: ExclusionReasons): Unit = {
    er1.name shouldBe er2.name
    er1.trainingUnfilledState shouldBe er2.trainingUnfilledState
    er1.trainingNullLabelLeaker shouldBe er2.trainingNullLabelLeaker
    er1.scoringUnfilledState shouldBe er2.scoringUnfilledState
    er1.jsDivergenceMismatch shouldBe er2.jsDivergenceMismatch
    er1.fillRateDiffMismatch shouldBe er2.fillRateDiffMismatch
    er1.fillRatioDiffMismatch shouldBe er2.fillRatioDiffMismatch
    er1.excluded shouldBe er2.excluded
  }

  def compareSeqExclusionReasons(er1: Seq[ExclusionReasons], er2: Seq[ExclusionReasons]): Unit = {
    er1.zip(er2).foreach { case (a, b) => compareExclusionReasons(a, b) }
  }

  def compare(rff1: RawFeatureFilterResults, rff2: RawFeatureFilterResults): Unit = {
    compareConfig(rff1.rawFeatureFilterConfig, rff2.rawFeatureFilterConfig)
    compareSeqDistributions(rff1.rawFeatureDistributions, rff2.rawFeatureDistributions)
    compareSeqMetrics(rff1.rawFeatureFilterMetrics, rff2.rawFeatureFilterMetrics)
    compareSeqExclusionReasons(rff1.exclusionReasons, rff2.exclusionReasons)
  }
}
