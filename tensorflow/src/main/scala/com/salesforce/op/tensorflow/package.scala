package com.salesforce.op

import java.nio.{DoubleBuffer, FloatBuffer, IntBuffer, LongBuffer}

import org.bytedeco.tensorflow._

package object tensorflow {

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

    def asCharArray: Array[Char] = asString.toCharArray

  }


}
