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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DataSplitterTest extends FlatSpec with TestSparkContext {
  import spark.implicits._

  val seed = 1234L
  val dataCount = 1000

  val data =
    RandomRDDs.normalVectorRDD(sc, 1000, 3, seed = seed)
      .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDS()

  val dataSplitter = new DataSplitter().setSeed(seed)

  Spec[DataSplitter] should "split the data in the appropriate proportion" in {
    val (train0, test0) = dataSplitter.setReserveTestFraction(0.0).split(data)
    test0.count() shouldBe 0
    train0.count() shouldBe dataCount

    val (train2, test2) = dataSplitter.setReserveTestFraction(0.2).split(data)
    math.abs(test2.count() - 200) < 30 shouldBe true
    math.abs(train2.count() - 800) < 30 shouldBe true

    val (train6, test6) = dataSplitter.setReserveTestFraction(0.6).split(data)
    math.abs(test6.count() - 600) < 30 shouldBe true
    math.abs(train6.count() - 400) < 30 shouldBe true
  }

  it should "wrap the data in the the ModelData class without changing it when prepare is called" in {
    val train = dataSplitter.prepare(data)
    train.train.collect().zip(data.collect()).foreach{ case (a, b) => a shouldBe b }
  }

}
