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

package com.salesforce.op

import java.nio.{DoubleBuffer, FloatBuffer, IntBuffer, LongBuffer}

import org.bytedeco.tensorflow._
import org.bytedeco.tensorflow.global.tensorflow._

package object tensorflow {

  /**
   * Enrichment for [[Tensor]] type to allow extracting values out
   *
   * @param t [[Tensor]] instance
   */
  implicit class RichTensor(val t: Tensor) extends AnyVal {

    def asIntArray: Array[Int] = {
      val res = new Array[Int](t.NumElements().toInt)
      t.createBuffer[IntBuffer]().get(res)
      res
    }

    def asLongArray: Array[Long] = {
      val res = new Array[Long](t.NumElements().toInt)
      t.createBuffer[LongBuffer]().get(res)
      res
    }

    def asFloatArray: Array[Float] = {
      val res = new Array[Float](t.NumElements().toInt)
      t.createBuffer[FloatBuffer]().get(res)
      res
    }

    def asDoubleArray: Array[Double] = {
      val res = new Array[Double](t.NumElements().toInt)
      t.createBuffer[DoubleBuffer]().get(res)
      res
    }

    def asString: String = t.createStringArray().toString

  }

  /**
   * Enrichment to handle TensorFlow status values
   *
   * @param s [[Status]] instance
   */
  implicit class RichStatus(val s: Status) extends AnyVal {

    /**
     * Checks if [[Status]].code == OK, otherwise throws [[RuntimeException]]
     * @throws RuntimeException is [[Status]].code != OK
     */
    def errorIfNotOK(): Unit = if (s.code() != OK) throw new RuntimeException(s.error_message().getString)

  }

  /**
   * Enrichment for [[Float]] value conversion
   *
   * @param f [[Float]] instance
   */
  implicit class RichFloatForTensorFlow(val f: Float) extends AnyVal {

    def asTensor: Tensor = {
      val tensor = new Tensor(DT_FLOAT, new TensorShape(1))
      tensor.createBuffer[FloatBuffer]().put(f)
      tensor
    }
  }

}
