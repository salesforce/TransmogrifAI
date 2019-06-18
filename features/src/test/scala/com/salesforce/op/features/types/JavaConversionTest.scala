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

package com.salesforce.op.features.types

import java.util

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class JavaConversionTest extends FlatSpec with TestCommon {

  Spec[JavaConversionTest] should "convert java Map to TextMap" in {
    type T = util.HashMap[String, String]
    null.asInstanceOf[T].toTextMap shouldEqual TextMap(Map())
    val j = new T()
    j.toTextMap shouldEqual TextMap(Map())
    j.put("A", "a")
    j.toTextMap shouldEqual TextMap(Map("A" -> "a"))
  }

  it should "convert java Map to MultiPickListMap" in {
    type T = util.HashMap[String, java.util.HashSet[String]]
    null.asInstanceOf[T].toMultiPickListMap shouldEqual MultiPickListMap(Map())
    val j = new T()
    j.toMultiPickListMap shouldEqual MultiPickListMap(Map())
    val h = new util.HashSet[String]()
    h.add("X")
    h.add("Y")
    j.put("test", h)
    j.toMultiPickListMap shouldEqual MultiPickListMap(Map("test" -> Set("X", "Y")))
  }

  it should "convert java Map to IntegralMap" in {
    type T = util.HashMap[String, java.lang.Long]
    null.asInstanceOf[T].toIntegralMap shouldEqual IntegralMap(Map())
    val j = new T()
    j.toIntegralMap shouldEqual IntegralMap(Map())
    j.put("test", java.lang.Long.valueOf(17))
    j.toIntegralMap shouldEqual IntegralMap(Map("test" -> 17))
  }

  it should "convert java Map to RealMap" in {
    type T = util.HashMap[String, java.lang.Double]
    null.asInstanceOf[T].toRealMap shouldEqual RealMap(Map())
    val j = new T()
    j.toRealMap shouldEqual RealMap(Map())
    j.put("test", java.lang.Double.valueOf(17.5))
    j.toRealMap shouldEqual RealMap(Map("test" -> 17.5))
  }

  it should "convert java Map to BinaryMap" in {
    type T = util.HashMap[String, java.lang.Boolean]
    null.asInstanceOf[T].toBinaryMap shouldEqual RealMap(Map())
    val j = new T()
    j.toBinaryMap shouldEqual RealMap(Map())
    j.put("test1", java.lang.Boolean.TRUE)
    j.put("test0", java.lang.Boolean.FALSE)
    j.toBinaryMap shouldEqual BinaryMap(Map("test1" -> true, "test0" -> false))
  }

}
