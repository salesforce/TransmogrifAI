/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.reflection


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

  it should "return a dealiased TypeTag for an type alias" in {
    val tTag = typeTag[scala.collection.immutable.List[String]]
    val aliasTag = typeTag[ListStringAlias]
    val dealiasedTag = ReflectionUtils.dealiasedTypeTag[ListStringAlias]

    tTag should not be equal(aliasTag)
    tTag.tpe shouldBe aliasTag.tpe.dealias
    tTag shouldBe dealiasedTag
    dealiasedTag.tpe shouldBe aliasTag.tpe.dealias
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

  it should "create a new instance of a class" in {
    val ctorArgs = (argName: String, argSymbol: Symbol) => Try {
      argName match {
        case "x" => 1
        case "y" => 2
      }
    }
    val instance = ReflectionUtils.newInstance[TestValCaseClass](classOf[TestValCaseClass], ctorArgs)
    instance.x shouldBe 1
    instance.y shouldBe 2
  }

  it should "fail to copy an instance that has a private ctor argument" in {
    val orig = new TestShouldFailClass(x = 1, y = "private ctor arg")
    an[RuntimeException] should be thrownBy ReflectionUtils.copy(orig)
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

}
