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

package com.salesforce.op.cli.gen

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks


@RunWith(classOf[JUnitRunner])
class FileGeneratorTest extends PropSpec with TestCommon with PropertyChecks {

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
    FileGenerator.skipWhitespace(sourceLeft, sourceLeft.lastIndexOf('/') + 1, +1) should equal(sourceLeft.indexOf('w'))
    FileGenerator.skipWhitespace(sourceRight, sourceRight.indexOf('/') - 1, -1) should equal(sourceRight.indexOf('d'))
  }

  property("skips expressions") {
    FileGenerator.skipExpr(bigSourceLeft, bigSourceLeft.indexOf('{'), +1) should equal(
      bigSourceLeft.lastIndexOf('}') + 1
    )
    FileGenerator.skipExpr(bigSourceRight, bigSourceRight.lastIndexOf('}'), -1) should equal(
      bigSourceRight.indexOf('{') - 1
    )
  }

  property("finds the right amount of substitution text in template") {
    FileGenerator.getSubstitutionEnd(bigSourceLeft, bigSourceLeft.lastIndexOf('/') + 1, +1) should equal(
      bigSourceLeft.lastIndexOf('}') + 1
    )
    FileGenerator.getSubstitutionEnd(bigSourceRight, bigSourceRight.indexOf('/') - 1, -1) should equal(
      bigSourceRight.indexOf('{') - 1
    )
  }

}
