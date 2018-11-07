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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.stages.impl.selector.ModelSelectorNames
import org.scalatest.{Assertion, Matchers}

/**
 * Assertion helpers for [[SplitterSummary]] child classes
 */
trait SplitterSummaryAsserts {
  self: Matchers =>

  def assertDataBalancerSummary(summary: Option[SplitterSummary])(
    assert: DataBalancerSummary => Assertion
  ): Assertion = summary match {
    case Some(s: DataBalancerSummary) =>
      val meta = s.toMetadata()
      meta.getString(SplitterSummary.ClassName) shouldBe classOf[DataBalancerSummary].getName
      meta.getLong(ModelSelectorNames.Positive) should be >= 0L
      meta.getLong(ModelSelectorNames.Negative) should be >= 0L
      meta.getDouble(ModelSelectorNames.Desired) should be >= 0.0
      meta.getDouble(ModelSelectorNames.UpSample) should be >= 0.0
      meta.getDouble(ModelSelectorNames.DownSample) should be >= 0.0
      assert(s)
    case x =>
      fail(s"Unexpected data balancer summary: $x")
  }

  def assertDataCutterSummary(summary: Option[SplitterSummary])(
    assert: DataCutterSummary => Assertion
  ): Assertion = summary match {
    case Some(s: DataCutterSummary) =>
      val meta = s.toMetadata()
      meta.getString(SplitterSummary.ClassName) shouldBe classOf[DataCutterSummary].getName
      meta.getDoubleArray(ModelSelectorNames.LabelsKept).foreach(_ should be >= 0.0)
      meta.getDoubleArray(ModelSelectorNames.LabelsDropped).foreach(_ should be >= 0.0)
      assert(s)
    case x =>
      fail(s"Unexpected data cutter summary: $x")
  }

  def assertDataSplitterSummary(summary: Option[SplitterSummary])(
    assert: DataSplitterSummary => Assertion
  ): Assertion = summary match {
    case Some(s: DataSplitterSummary) =>
      val meta = s.toMetadata()
      meta.getString(SplitterSummary.ClassName) shouldBe classOf[DataSplitterSummary].getName
      assert(s)
    case x =>
      fail(s"Unexpected data splitter summary: $x")
  }

}
