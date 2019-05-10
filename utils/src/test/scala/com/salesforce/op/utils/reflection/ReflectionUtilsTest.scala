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

package com.salesforce.op.utils.reflection


import com.salesforce.op.utils.types.TestPrivateType
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.runtime.universe._
import scala.util.Try

trait TestBase[T] {
  def t: T
  def func: Int => Boolean
  def ttag: TypeTag[T]
}

class TestValClass(x: Int, y: Int)

case class TestValCaseClass(x: Int, y: Int)

class TestClass[T]
(
  val i: Int,
  val l: Long,
  val d: Double,
  val f: Float,
  val s: Short,
  val b: Byte,
  val c: Char,
  val bool: Boolean,
  val func: Int => Boolean,
  val str: String,
  val opt: Option[String],
  val ls: List[String],
  val st: Set[String],
  val m: Map[String, Int],
  val t: T,
  val tc: TestValCaseClass
)(implicit val ttag: TypeTag[T]) extends TestBase[T]

class TestShouldFailClass(val x: Int, y: String)

class TestClassNoArgs() {
  val x = 123
}

object TestObject {
  val x = 456
}

class TestClassVar {
  var myVar: Option[String] = None
  def setMyVar(s: String): this.type = {
    myVar = Option(s)
    this
  }
  private def getValue: Int = 2
  def getValuePerf: Int = 2

  def boo(x: Int, y: Int): Int = boo(x + y)
  def boo(x: Int): Int = x
  def boo(): Int = boo(1)
}

@RunWith(classOf[JUnitRunner])
class ReflectionUtilsTest extends FlatSpec with Matchers {

  type ListStringAlias = scala.collection.immutable.List[String]

  "ReflectionUtils" should "return a runtime mirror" in {
    ReflectionUtils.runtimeMirror() shouldBe a[Mirror]
    ReflectionUtils.runtimeMirror(this.getClass.getClassLoader) shouldBe a[Mirror]
  }

  it should "return a TypeTag of a type" in {
    val ttag = typeTag[List[String]]
    val resttag = ReflectionUtils.typeTagForType[List[String]](tpe = ttag.tpe)
    resttag shouldBe ttag
    resttag.tpe =:= ttag.tpe shouldBe true
  }

  it should "return a TypeTag of a type name" in {
    val ttag = typeTag[TestValClass]
    val typeName = ttag.tpe.typeSymbol.fullName
    val resttag = ReflectionUtils.typeTagForTypeName[TestValClass](typeName)
    resttag shouldBe ttag
    resttag.tpe =:= ttag.tpe shouldBe true
  }

  it should "return a dealiased TypeTag for an type alias" in {
    val tTag = typeTag[scala.collection.immutable.List[String]]
    val aliasTag = typeTag[ListStringAlias]
    val dealiasedTag = ReflectionUtils.dealiasedTypeTagForType[ListStringAlias]()

    tTag should not be equal(aliasTag)
    tTag.tpe shouldBe aliasTag.tpe.dealias
    tTag shouldBe dealiasedTag
    dealiasedTag.tpe shouldBe aliasTag.tpe.dealias
  }

  it should "deep dealias types" in {
    val tt = typeTag[Map[String, Seq[(Double, ListStringAlias)]]].tpe
    ReflectionUtils.dealisedTypeName(tt) shouldBe
      "scala.collection.immutable.Map[" +
        "java.lang.String," +
        "scala.collection.Seq[scala.Tuple2[scala.Double,scala.collection.immutable.List[java.lang.String]]]]"
  }

  it should "allow copying a class" in {
    val orig = new TestClass[TestValClass](
      i = 123,
      l = 456L,
      d = 1.2345,
      f = 575f,
      s = 3,
      b = 123,
      c = 'c',
      bool = true,
      func = _ % 2 == 0,
      str = "test string",
      opt = Option("boo"),
      ls = List("2", "3"),
      st = Set("a", "b", "c"),
      m = Map("one" -> 1, "two" -> 2, "three" -> 3),
      t = new TestValClass(x = 123, y = 456),
      tc = TestValCaseClass(x = 777, y = 888)
    )
    val copy = ReflectionUtils.copy(orig)

    orig.i shouldBe copy.i
    orig.l shouldBe copy.l
    orig.d shouldBe copy.d
    orig.f shouldBe copy.f
    orig.s shouldBe copy.s
    orig.b shouldBe copy.b
    orig.c shouldBe copy.c
    orig.bool shouldBe copy.bool
    orig.func shouldBe copy.func
    orig.str shouldBe copy.str
    orig.opt shouldBe copy.opt
    orig.ls shouldBe copy.ls
    orig.st shouldBe copy.st
    orig.m shouldBe copy.m
    orig.t shouldBe copy.t
    orig.tc shouldBe copy.tc
    orig.ttag shouldBe copy.ttag
  }

  it should "allow copying a class with a private package ctor" in {
    val orig = TestPrivateType(1, 2)
    val copy = ReflectionUtils.copy(orig)
    orig.x shouldBe copy.x
    orig.y shouldBe copy.y
  }

  it should "find a private package ctor correctly with args" in {
    val (_, argsList) = ReflectionUtils.bestCtorWithArgs(TestPrivateType(1, 2))
    argsList shouldBe List("x" -> 1, "y" -> 2)
  }

  it should "create a new instance of a class" in {
    val ctorArgs = (argName: String, argSymbol: Symbol) => Try { argName match { case "x" => 1; case "y" => 2 } }
    val instance = ReflectionUtils.newInstance[TestValCaseClass](classOf[TestValCaseClass], ctorArgs)
    instance.x shouldBe 1
    instance.y shouldBe 2
  }

  it should "create a new instance by a class name" in {
    val instance = ReflectionUtils.newInstance[TestClassNoArgs](classOf[TestClassNoArgs].getName)
    instance.x shouldBe 123
  }

  it should "return object instance by its class name" in {
    val instance = ReflectionUtils.newInstance[TestObject.type](TestObject.getClass.getName)
    instance.x shouldBe 456
  }

  it should "fail to create a class if neither args ctor is not present nor it's an object" in {
    a[RuntimeException] should be thrownBy
      ReflectionUtils.newInstance[TestShouldFailClass](classOf[TestShouldFailClass].getName)
  }

  it should "fail to copy an instance that has a private ctor argument" in {
    val orig = new TestShouldFailClass(x = 1, y = "private ctor arg")
    a[RuntimeException] should be thrownBy ReflectionUtils.copy(orig)
  }

  it should "return a class for name" in {
    val klazz = classOf[List[String]]
    val res = ReflectionUtils.classForName(classOf[List[String]].getName)
    klazz shouldBe res
  }

  it should "create a manifest for a type tag" in {
    val m = ReflectionUtils.manifestForTypeTag[Map[String, Option[Long]]]
    m.toString() shouldBe "scala.collection.immutable.Map[java.lang.String, scala.Option[long]]"
  }

  it should "create a class tag from a weak type tag" in {
    val c = ReflectionUtils.classTagForWeakTypeTag[Map[String, Option[Long]]]
    c.toString() shouldBe "scala.collection.immutable.Map"
  }

  it should "allow you to find and use a setter for a class" in {
    val myClass = new TestClassVar()
    val setter = ReflectionUtils.reflectSetterMethod(myClass, "myVar", Seq("yay"))
    myClass.myVar shouldBe Some("yay")
  }

  it should "allow you to find and use a private method for a class" in {
    val myClass = new TestClassVar()
    val value = ReflectionUtils.reflectMethod(myClass, "getValue").apply()
    value shouldBe 2
  }

  it should "reflected method should be fast to execute" in {
    val myClass = new TestClassVar()
    val method = ReflectionUtils.reflectMethod(myClass, "getValue")
    val max = 100000
    def measure(fun: => Int): Long = {
      val start = System.currentTimeMillis()
      (0 until max).foreach(_ => fun shouldBe 2)
      System.currentTimeMillis() - start
    }
    val warmUp = measure(method.apply().asInstanceOf[Int]) -> measure(myClass.getValuePerf) // warm up
    val elapsedReflect = measure(method.apply().asInstanceOf[Int])
    val actual = measure(myClass.getValuePerf)

    elapsedReflect should be <= 10 * actual
  }

  it should "error on reflecting a non existent method" in {
    val myClass = new TestClassVar()
    val err = intercept[RuntimeException](ReflectionUtils.reflectMethod(myClass, "non_existent"))
    err.getMessage shouldBe
      s"Method with name 'non_existent' was not found on instance of type: ${myClass.getClass}"
  }

  it should "reflect methods with largest number of arguments by default" in {
    val myClass = new TestClassVar()
    val boo = ReflectionUtils.reflectMethod(myClass, "boo", argsCount = None)
    boo(2, 3) shouldBe 5
  }

  it should "reflect methods with various number of arguments" in {
    val myClass = new TestClassVar()
    val boo = ReflectionUtils.reflectMethod(myClass, "boo", argsCount = Some(0))
    val boo1 = ReflectionUtils.reflectMethod(myClass, "boo", argsCount = Some(1))
    val boo2 = ReflectionUtils.reflectMethod(myClass, "boo", argsCount = Some(2))
    boo() shouldBe 1
    boo1(2) shouldBe 2
    boo2(2, 3) shouldBe 5
  }

}

