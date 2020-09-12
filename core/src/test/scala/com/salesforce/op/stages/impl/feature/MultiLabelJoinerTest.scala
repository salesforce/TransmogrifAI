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
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.reflect.ClassTag

@RunWith(classOf[JUnitRunner])
class MultiLabelJoinerTest extends MultiLabelJoinerBaseTest[MultiLabelJoiner] {
  val transformer = new MultiLabelJoiner().setInput(classIndexFeature, probVecFeature)

  val expectedResult = Seq(
    classes.zip(Array(40.0, 30.0, 20.0, 0.0)).toMap.toRealMap,
    classes.zip(Array(20.0, 40.0, 30.0, 0.0)).toMap.toRealMap,
    classes.zip(Array(30.0, 20.0, 40.0, 0.0)).toMap.toRealMap
  )
}

abstract class MultiLabelJoinerBaseTest[T <: MultiLabelJoiner : ClassTag] extends OpTransformerSpec[RealMap, T] {
  // Input Dataset and features
  val (inputDF, idFeature, classFeature, probVecFeature) = TestFeatureBuilder("ID", "class", "prob",
    Seq[(Integral, Text, OPVector)](
      (Integral(1001), Text("Low"), OPVector(Vectors.dense(Array(40.0, 30.0, 20.0, 0.0)))),
      (Integral(1002), Text("Medium"), OPVector(Vectors.dense(Array(20.0, 40.0, 30.0, 0.0)))),
      (Integral(1003), Text("High"), OPVector(Vectors.dense(Array(30.0, 20.0, 40.0, 0.0))))
    )
  )
  val classIndexFeature = classFeature.indexed(unseenName = OpStringIndexerNoFilter.UnseenNameDefault)

  // String indexer stage estimator.
  val indexStage = classIndexFeature.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(inputDF)
  val inputData = indexStage.transform(inputDF)

  // Apart from classes in the data - Low, High, Medium, there is an additional class - UnseenLabel for unseen classes.
  val classes = indexStage.getMetadata().getMetadata("ml_attr").getStringArray("vals")
}
