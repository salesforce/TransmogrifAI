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
package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpRandomForestClassifier, OpXGBoostClassifier}
import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RandomParamBuilderTest extends FlatSpec with TestSparkContext {

  private val lr = new OpLogisticRegression()
  private val rf = new OpRandomForestClassifier()
  private val xgb = new OpXGBoostClassifier()


  it should "build a param grid of the desired length with one param variable" in {
    val min = 0.00001
    val max = 10
    val lrParams = new RandomParamBuilder()
      .uniform(lr.regParam, min, max)
      .build(5)
    lrParams.length shouldBe 5
    lrParams.foreach(_.toSeq.length shouldBe 1)
    lrParams.foreach(_.toSeq.foreach( p => (p.value.asInstanceOf[Double] < max &&
      p.value.asInstanceOf[Double] > min) shouldBe true))
    lrParams.foreach(_.toSeq.map(_.param).toSet shouldBe Set(lr.regParam))

    val lrParams2 = new RandomParamBuilder()
      .exponential(lr.regParam, min, max)
      .build(20)
    lrParams2.length shouldBe 20
    lrParams2.foreach(_.toSeq.length shouldBe 1)
    lrParams2.foreach(_.toSeq.foreach( p => (p.value.asInstanceOf[Double] < max &&
      p.value.asInstanceOf[Double] > min) shouldBe true))
    lrParams2.foreach(_.toSeq.map(_.param).toSet shouldBe Set(lr.regParam))
  }

  it should "build a param grid of the desired length with many param variables" in {
    val lrParams = new RandomParamBuilder()
      .exponential(lr.regParam, .000001, 10)
      .subset(lr.family, Seq("auto", "binomial", "multinomial"))
      .uniform(lr.maxIter, 2, 50)
      .build(23)
    lrParams.length shouldBe 23
    lrParams.foreach(_.toSeq.length shouldBe 3)
    lrParams.foreach(_.toSeq.map(_.param).toSet shouldBe Set(lr.regParam, lr.family, lr.maxIter))
  }

  it should "work for all param types" in {
    val xgbParams = new RandomParamBuilder()
      .subset(xgb.checkpointPath, Seq("a", "b"))//string
      .uniform(xgb.alpha, 0, 1)//double
      .uniform(xgb.missing, 0, 100)//float
      .uniform(xgb.checkpointInterval, 2, 5)//int
      .uniform(xgb.seed, 5, 1000)//long
      .uniform(xgb.useExternalMemory)//boolean
      .exponential(xgb.baseScore, 0.0001, 1)//double
      .exponential(xgb.missing, 0.000001F, 1) //float - overwrites first call
      .build(2)

    xgbParams.length shouldBe 2
    xgbParams.foreach(_.toSeq.length shouldBe 7)
    xgbParams.foreach(_.toSeq.map(_.param).toSet shouldBe Set(xgb.checkpointPath, xgb.alpha, xgb.missing,
      xgb.checkpointInterval, xgb.seed, xgb.useExternalMemory, xgb.baseScore))
  }

  it should "throw an assert error if an improper min value is passed in for exponential scale" in {
    intercept[AssertionError]( new RandomParamBuilder()
      .exponential(xgb.baseScore, 0, 1)).getMessage() shouldBe
      "assertion failed: Min value must be greater than zero for exponential distribution to work"
  }

}
