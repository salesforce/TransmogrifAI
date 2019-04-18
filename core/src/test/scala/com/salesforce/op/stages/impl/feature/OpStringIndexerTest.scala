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

import com.salesforce.op.features.types._
import com.salesforce.op.features.types.Text
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class OpStringIndexerTest extends FlatSpec with TestSparkContext {

  val txtData = Seq("a", "b", "c", "a", "a", "c").map(_.toText)
  val (ds, txtF) = TestFeatureBuilder(txtData)
  val expected = Array(0.0, 2.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN)


  Spec[OpStringIndexer[_]] should "correctly set the wrapped spark stage params" in {
    val indexer = new OpStringIndexer[Text]()
    indexer.setHandleInvalid(StringIndexerHandleInvalid.Skip)
    indexer.getSparkMlStage().get.getHandleInvalid shouldBe StringIndexerHandleInvalid.Skip.entryName.toLowerCase
    indexer.setHandleInvalid(StringIndexerHandleInvalid.Error)
    indexer.getSparkMlStage().get.getHandleInvalid shouldBe StringIndexerHandleInvalid.Error.entryName.toLowerCase
    indexer.setHandleInvalid(StringIndexerHandleInvalid.Keep)
    indexer.getSparkMlStage().get.getHandleInvalid shouldBe StringIndexerHandleInvalid.Keep.entryName.toLowerCase
  }

  it should "throw an error if you try to set noFilter as the indexer" in {
    val indexer = new OpStringIndexer[Text]()
    intercept[IllegalArgumentException](indexer.setHandleInvalid(StringIndexerHandleInvalid.NoFilter))
  }

  it should "correctly index a text column" in {
    val stringIndexer = new OpStringIndexer[Text]().setInput(txtF)
    val indices = stringIndexer.fit(ds).transform(ds).collect(stringIndexer.getOutput())

    indices shouldBe expected
  }

  it should "correctly deindex a numeric column" in {
    val indexedStage = new OpStringIndexer[Text]().setInput(txtF)
    val indexed = indexedStage.getOutput()
    val indices = indexedStage.fit(ds).transform(ds)
    val deindexedStage = new OpIndexToString().setInput(indexed)
    val deindexed = deindexedStage.getOutput()
    val deindexedData = deindexedStage.transform(indices).collect(deindexed)
    deindexedData shouldBe txtData
  }

}
