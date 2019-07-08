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

import org.bytedeco.tensorflow._
import org.bytedeco.tensorflow.global.tensorflow._

/**
 * Just a simple TensorFlow model used in test that computes matrices filled in with value 27.
 *
 * It creates a graph of three matrices of specified sizes filled with:
 * ones, sixes and tens respectively, then it sums them.
 *
 * Result should be a matrix of specified sizes with value 27 in all cells.
 *
 * Example for: new SimpleTensorFlowModel(2).run() should yield [27, 27]
 *
 * @param sizes matrix dimensions
 */
class All27Model(sizes: Long*) {

  def graph(): GraphDef = {
    // Create a new empty graph
    val scope = Scope.NewRootScope

  // Matrices of ones, sixes and tens in specified sizes
  val shape = new TensorShape(sizes: _*)
  val ones = Const(scope.WithOpName("ones"), 1, shape)
  val sixes = Const(scope.WithOpName("sixes"), 6, shape)
  val tens = Const(scope.WithOpName("tens"), 20, shape)

    // Adding all matrices element-wise
    val ov = new OutputVector(ones, sixes, tens)
    val inputList = new InputList(ov)
    val add = new AddN(scope.WithOpName("add"), inputList)

    // Build a graph definition object
    val graph = new GraphDef
    scope.ToGraphDef(graph).errorIfNotOK()
    graph
  }

  def run(g: GraphDef = graph(), sessionOptions: SessionOptions = new SessionOptions): TensorVector = {
    val session = new Session(sessionOptions)
    try {
      session.Create(g).errorIfNotOK()
      val outputs = new TensorVector
      session.Run(new StringTensorPairVector, new StringVector("add"), new StringVector, outputs).errorIfNotOK()
      outputs
    } finally if (session != null) session.close()
  }

}
