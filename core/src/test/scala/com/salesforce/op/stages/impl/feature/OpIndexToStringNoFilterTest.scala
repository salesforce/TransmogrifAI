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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpIndexToStringNoFilterTest extends FlatSpec with TestSparkContext {

  val (ds, indF) = TestFeatureBuilder(Seq(0.0, 2.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN))
  val labels = Array("a", "c", "b")
  val expected = Array("a", "b", "c", "a", "a", "c").map(_.toText)

  val labelsNew = Array("a", "c")
  val expectedNew = Array("a", OpIndexToStringNoFilter.unseenDefault, "c", "a", "a", "c").map(_.toText)

  Spec[OpIndexToStringNoFilter] should "correctly deindex a numeric column" in {
    val indexToStr = new OpIndexToStringNoFilter().setInput(indF).setLabels(labels)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expected
  }

  it should "correctly deindex a numeric column (shortcut)" in {
    val str = indF.deindexed(labels)
    val strs = str.originStage.asInstanceOf[OpIndexToStringNoFilter].transform(ds).collect(str)
    strs shouldBe expected

    val str2 = indF.deindexed(labels, handleInvalid = IndexToStringHandleInvalid.Error)
    val strs2 = str2.originStage.asInstanceOf[OpIndexToString].transform(ds).collect(str2)
    strs2 shouldBe expected
  }

  it should "correctly deindex even if the labels list does not match the number of indicies" in {
    val indexToStr = new OpIndexToStringNoFilter().setInput(indF).setLabels(labelsNew)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expectedNew
  }

  Spec[OpIndexToString] should "correctly deindex a numeric column" in {
    val indexToStr = new OpIndexToString().setInput(indF).setLabels(labels)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expected
  }
}
