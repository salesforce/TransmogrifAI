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

package com.salesforce.op.tensorflow

import java.io.{File, FileOutputStream}

import com.salesforce.op.test.TestCommon
import org.bytedeco.tensorflow.{Session, SessionOptions}
import org.bytedeco.tensorflow.global.tensorflow._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterAll, FlatSpec}


@RunWith(classOf[JUnitRunner])
class TensorFlowModelTest extends FlatSpec with TestCommon with BeforeAndAfterAll {

  private lazy val graphData: Array[Byte] = {
    // Sample model copied from - https://github.com/tensorflow/models/tree/master/samples/languages/java/training
    val stream = getClass.getResourceAsStream("/model/graph.pb")
    val res = new Array[Byte](stream.available())
    stream.read(res)
    res
  }
  private lazy val graphFile: File = {
    val file = File.createTempFile("graph", ".pb")
    val out = new FileOutputStream(file)
    out.write(graphData)
    out.close()
    file
  }

  override def beforeAll(): Unit = {
    // Platform-specific initialization routine
    InitMain("trainer", null.asInstanceOf[Array[Int]], null)
    graphFile.deleteOnExit()
  }

  override def afterAll(): Unit = {
    if (graphFile.exists()) graphFile.delete()
  }

  Spec[All27Model] should "create a graph" in {
    noException shouldBe thrownBy (new All27Model().graph())
  }

  it should "run" in {
    val tfModel = new All27Model(2, 3)
    val result = tfModel.run()

    result.get() match {
      case Array(v) => v.asIntArray shouldBe Array.fill(6)(27)
      case v => fail("Unexpected result of size " + v.length)
    }
  }

  Spec[SimpleLinearModel] should "create a graph" in {
    noException shouldBe thrownBy (new SimpleLinearModel(graphFile.getAbsolutePath).graph())
  }

  it should "train & predict" in {
    val tfModel = new SimpleLinearModel(graphFile.getAbsolutePath)
    implicit val session: Session = new Session(new SessionOptions())

    try {
      // Train.
      val trainingData = (0 until 2500).map(_ => util.Random.nextFloat()).map(x => x -> (3f * x + 2f))
      tfModel.train()(trainingData)

      tfModel.getW shouldBe (3f +- .2f)
      tfModel.getB shouldBe (2f +- .2f)

      // Predict.
      // Ideally would produce: 3 * x + 2
      for {
        (x, prediction) <- (0 until 10).map(x => x -> tfModel.predict(x))
      } prediction.get() match {
        case Array(v) if v.NumElements() == 1 => v.asFloatArray(0) shouldBe ((3f * x + 2f) +- .5f)
        case v => fail("Unexpected result of size " + v.length)
      }
    } finally if (session != null) session.close()

  }

}
