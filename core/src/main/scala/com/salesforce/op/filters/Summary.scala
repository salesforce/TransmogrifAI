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

import com.salesforce.op.stages.impl.feature.TextStats
import com.twitter.algebird._

/**
 * Class used to get summaries of prepared features to determine distribution binning strategy
 *
 * @param min   minimum value seen for double, minimum number of tokens in one text for text
 * @param max   maximum value seen for double, maximum number of tokens in one text for text
 * @param sum   sum of values for double, total number of tokens for text
 * @param count number of doubles for double, number of texts for text
 * @param maxCardinality maximum number of unique tokens to keep track of, for a given text feature
 */
case class Summary(min: Double, max: Double, sum: Double, count: Double,
                   textLength: Option[Moments] = None,
                   textCard: Option[TextStats] = None,
                   maxCardinality: Int = 500)

case object Summary {

  val empty: Summary = Summary(Double.PositiveInfinity, Double.NegativeInfinity, 0.0, 0.0)

  implicit val monoid: Monoid[Summary] = new Monoid[Summary] {
    override def zero = Summary.empty
    override def plus(l: Summary, r: Summary) = {
      implicit val testStatsSG: Semigroup[TextStats] = TextStats.semiGroup(l.maxCardinality)
      val combinedtextLen: Option[Moments] = (l.textLength, r.textLength) match {
        case (Some(leftTL), Some(rightTL)) => Some(MomentsGroup.plus(leftTL, rightTL))
        case (Some(leftTL), None) => Some(leftTL)
        case (None, Some(rightTL)) => Some(rightTL)
        case _ => None
      }
      val combinedtextCard: Option[TextStats] = (l.textCard, r.textCard) match {
        case (Some(leftTC), Some(rightTC)) => Some(testStatsSG.plus(leftTC, rightTC))
        case (Some(leftTC), None) => Some(leftTC)
        case (None, Some(rightTC)) => Some(rightTC)
        case _ => None
      }
      Summary(
        math.min(l.min, r.min), math.max(l.max, r.max), l.sum + r.sum, l.count + r.count,
        combinedtextLen, combinedtextCard
      )
    }
  }

  /**
   * @param preppedFeature processed feature
   * @return feature summary derived from processed feature
   */
  def apply(preppedFeature: ProcessedSeq): Summary = {
    preppedFeature match {
      case Left(v) =>
        val textLenMoments = MomentsGroup.sum(v.map(x => Moments(x.length.toDouble)))
        val tokenDistribution = TextStats(v.groupBy(identity).mapValues(_.size))
        Summary(v.size, v.size, v.size, 1.0, Some(textLenMoments), Some(tokenDistribution))
      case Right(v) => monoid.sum(v.map(d => Summary(d, d, d, 1.0)))
    }
  }
}
