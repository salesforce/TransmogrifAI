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
import com.salesforce.op.stages.base.unary.UnaryModel
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid.Skip
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.feature.StringIndexerModel
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpStringIndexerNoFilterTest extends OpEstimatorSpec[RealNN, UnaryModel[Text, RealNN], OpStringIndexerNoFilter[Text]] {

  val txtData = Seq("a", "b", "c", "a", "a", "c").map(_.toText)
  val (inputData, txtF) = TestFeatureBuilder(txtData)
  override val expectedResult: Seq[RealNN] = Array(0.0, 2.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN)

  override val estimator: OpStringIndexerNoFilter[Text] = new OpStringIndexerNoFilter[Text]().setInput(txtF)

  val txtDataNew = Seq("a", "b", "c", "a", "a", "c", "d", "e").map(_.toText)
  val (dsNew, txtFNew) = TestFeatureBuilder(txtDataNew)
  val expectedNew = Array(0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0).map(_.toRealNN)

  it should "correctly index a text column (shortcut)" in {
    val indexed = txtF.indexed()
    val indices = indexed.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(inputData).transform(inputData).collect(indexed)
    indices shouldBe expectedResult

    val indexed2 = txtF.indexed(handleInvalid = Skip)
    val indicesfit = indexed2.originStage.asInstanceOf[OpStringIndexer[_]].fit(inputData)
    val indices2 = indicesfit.transform(inputData).collect(indexed2)
    val indices3 = indicesfit.asInstanceOf[SwUnaryModel[Text, RealNN, StringIndexerModel]]
      .setInput(txtFNew).transform(dsNew).collect(indexed2)
    indices2 shouldBe expectedResult
    indices3 shouldBe expectedResult
  }

  it should "correctly deinxed a numeric column" in {
    val indexed = txtF.indexed()
    val indices = indexed.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(inputData).transform(inputData)
    val deindexed = indexed.deindexed()
    val deindexedData = deindexed.originStage.asInstanceOf[OpIndexToStringNoFilter]
      .transform(indices).collect(deindexed)
    deindexedData shouldBe txtData
  }

  it should "assign new strings to the unseen string category" in {
    val indices = estimator.fit(inputData).setInput(txtFNew).transform(dsNew).collect(estimator.getOutput())
    indices shouldBe expectedNew
  }
}
