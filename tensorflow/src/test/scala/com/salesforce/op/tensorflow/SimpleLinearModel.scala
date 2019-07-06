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

import java.nio.FloatBuffer

import org.bytedeco.tensorflow._
import org.bytedeco.tensorflow.global.tensorflow._

/**
 * A very simple linear model: y = x * W + b
 * Copied from - https://github.com/tensorflow/models/tree/master/samples/languages/java/training
 *
 * Train on pairs (x,y)
 * Predict 'y' for given 'x'
 *
 * @param graphFile path to graph (.pb) to load
 */
class SimpleLinearModel(graphFile: String) {

  def graph(): GraphDef = {
    val graphDef = new GraphDef()
    ReadBinaryProto(Env.Default(), graphFile, graphDef).errorIfNotOK()
    graphDef
  }

  def train(session: Session, g: GraphDef = graph())(data: => Seq[(Float, Float)]): Unit = {
    session.Create(g).errorIfNotOK()

    // TODO: run in parallel, e.g.
    // https://github.com/bytedeco/javacpp-presets/blob/master/tensorflow/samples/ExampleTrainer.java#L142
    for {(x, y) <- data} {
      val input_feed = new StringTensorPairVector(
        Array("input", "target"), Array[Tensor](toTensor(x), toTensor(y))
      )
      val outputs = new TensorVector
      session.Run(input_feed, new StringVector, new StringVector("init:0", "train:0"), outputs).errorIfNotOK()
    }
  }

  def predict(session: Session, x: Float): TensorVector = {
    try {
      val input_feed = new StringTensorPairVector(Array("input"), Array[Tensor](toTensor(x)))
      val outputs = new TensorVector
      session.Run(input_feed, new StringVector("output:0"), new StringVector("init:0"), outputs).errorIfNotOK()
      outputs
    } finally if (session != null) session.close()
  }

  private def toTensor(f: Float): Tensor = {
    val tensor = new Tensor(DT_FLOAT, new TensorShape(1))
    tensor.createBuffer[FloatBuffer]().put(f)
    tensor
  }


}
