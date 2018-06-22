/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.filters

import com.twitter.algebird.Monoid

/**
 * Class used to get summaries of prepared features to determine distribution binning strategy
 *
 * @param min minimum value seen
 * @param max maximum value seen
 */
private[op] case class Summary(min: Double, max: Double)

private[op] case object Summary {

  val empty: Summary = Summary(Double.PositiveInfinity, Double.NegativeInfinity)

  implicit val monoid: Monoid[Summary] = new Monoid[Summary] {
    override def zero = empty
    override def plus(l: Summary, r: Summary) = Summary(math.min(l.min, r.min), math.max(l.max, r.max))
  }

  /**
   * @param preppedFeature processed feature
   * @return feature summary derived from processed feature
   */
  def apply(preppedFeature: ProcessedSeq): Summary = {
    preppedFeature match {
      case Left(v) => Summary(v.size, v.size)
      case Right(v) => monoid.sum(v.map(d => Summary(d, d)))
    }
  }
}
