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

trait FiltersTestData {

  protected val eps = 1E-2

  protected val trainSummaries = Seq(
    FeatureDistribution("A", None, 10, 1, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistribution("B", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistribution("C", Some("1"), 10, 1, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistribution("C", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistribution("D", Some("1"), 10, 9, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistribution("D", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty)
  )

  protected val scoreSummaries = Seq(
    FeatureDistribution("A", None, 10, 8, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistribution("B", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistribution("C", Some("1"), 10, 1, Array(0, 0, 10, 10, 0), Array.empty),
    FeatureDistribution("C", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistribution("D", Some("1"), 0, 0, Array(0, 0, 0, 0, 0), Array.empty),
    FeatureDistribution("D", Some("2"), 0, 0, Array(0, 0, 0, 0, 0), Array.empty)
  )
}
