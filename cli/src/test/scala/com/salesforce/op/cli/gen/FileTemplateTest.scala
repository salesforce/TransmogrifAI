/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Matchers, PropSpec}


@RunWith(classOf[JUnitRunner])
class FileTemplateTest extends PropSpec with TestCommon with PropertyChecks {

  private val sourceLeft = "/* COMM */   \t \n  word"
  private val sourceRight = "word   \t \n    /* COMM */"

  private val bigSourceLeft =
    """/* COMM */   {
      |  val x = 3
      |  val y = 4
      |  def myMethod[X](myT: (X, X)): Option[X] = { {
      |    Some(myT._1 + 5)
      |  } }
      |} ( more code )
    """.stripMargin

  private val bigSourceRight =
    """
      |( more code ) {
      |  val x = 3
      |  val y = 4
      |  def myMethod[X](myT: (X, X)): Option[X] = { {
      |    Some(myT._1 + 5)
      |  } }
      |}   /* COMM */
    """.stripMargin

  property("skips whitespace") {
    FileTemplate.skipWhitespace(sourceLeft, sourceLeft.lastIndexOf('/') + 1, +1) should equal(sourceLeft.indexOf('w'))
    FileTemplate.skipWhitespace(sourceRight, sourceRight.indexOf('/') - 1, -1) should equal(sourceRight.indexOf('d'))
  }

  property("skips expressions") {
    FileTemplate.skipExpr(bigSourceLeft, bigSourceLeft.indexOf('{'), +1) should equal(
      bigSourceLeft.lastIndexOf('}') + 1
    )
    FileTemplate.skipExpr(bigSourceRight, bigSourceRight.lastIndexOf('}'), -1) should equal(
      bigSourceRight.indexOf('{') - 1
    )
  }

  property("finds the right amount of substitution text in template") {
    FileTemplate.getSubstitutionEnd(bigSourceLeft, bigSourceLeft.lastIndexOf('/') + 1, +1) should equal(
      bigSourceLeft.lastIndexOf('}') + 1
    )
    FileTemplate.getSubstitutionEnd(bigSourceRight, bigSourceRight.indexOf('/') - 1, -1) should equal(
      bigSourceRight.indexOf('{') - 1
    )
  }

}
