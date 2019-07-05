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
 * Just a simple TensorFlow model used in test.
 *
 * It creates a graph of three matrices of specified sizes filled with:
 * ones, sixes and tens respectively, then it sums them.
 *
 * Result should be a matrix of specified sizes with value 17 in all cells.
 *
 * @param sizes matrix dimensions
 */
class SimpleTensorFlowModel(sizes: Long*) {

  // Platform-specific initialization routine
  InitMain("trainer", null.asInstanceOf[Array[Int]], null)

  // Create a new empty graph
  val scope = Scope.NewRootScope

  // (2, 2) matrix of ones, sixes and tens
  val shape = new TensorShape(sizes: _*)
  val ones = Const(scope.WithOpName("ones"), 1, shape)
  val sixes = Const(scope.WithOpName("sixes"), 6, shape)
  val tens = Const(scope.WithOpName("tens"), 10, shape)

  // Adding all matrices element-wise
  val ov = new OutputVector(ones, sixes, tens)
  val inputList = new InputList(ov)
  val add = new AddN(scope.WithOpName("add"), inputList)

  // Build a graph definition object
  val graph = new GraphDef
  TF_CHECK_OK(scope.ToGraphDef(graph))

  // Creates a session.
  val sessionOptions = new SessionOptions
  val session = new Session(sessionOptions)

  def run(): TensorVector = {
    try { // Create the graph to be used for the session.
      TF_CHECK_OK(session.Create(graph))
      // Input and output of a single session run.
      val input_feed = new StringTensorPairVector
      val output_tensor_name = new StringVector("add:0")
      val target_tensor_name = new StringVector
      val outputs = new TensorVector
      // Run the session once
      TF_CHECK_OK(session.Run(input_feed, output_tensor_name, target_tensor_name, outputs))
      outputs
    } finally if (session != null) session.close()
  }

}
