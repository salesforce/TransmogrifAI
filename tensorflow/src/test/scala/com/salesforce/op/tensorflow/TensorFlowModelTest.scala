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
    }
  }

  Spec[SimpleLinearModel] should "create a graph" in {
    noException shouldBe thrownBy (new SimpleLinearModel(graphFile.getAbsolutePath).graph())
  }

  it should "run" in {
    val tfModel = new SimpleLinearModel(graphFile.getAbsolutePath)

    val result1 = tfModel.run(x = 11f, w = 5f, b = -2f)()
    result1.get() match {
      case Array(v) => v.asIntArray shouldBe Array(11f * 5f - 2f)
    }

    val result2 = tfModel.run(x = -2f, w = 3f, b = 100f)()
    result2.get() match {
      case Array(v) => v.asIntArray shouldBe Array(-2f * 3f + 100f)
    }
  }


}
