/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.table

import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.table.Alignment._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

case class Transaction(date: Long, amount: Double, source: String, status: String)

@RunWith(classOf[JUnitRunner])
class TableTest extends FlatSpec with TestCommon {

  val columns = Seq("date", "amount", "source", "status")
  val transactions = Seq(
    Transaction(1, 4.95, "Cafe Venetia", "Success"),
    Transaction(2, 12.65, "Sprout", "Success"),
    Transaction(3, 4.75, "Caltrain", "Pending")
  )

  Spec[Table[_]] should "error on missing columns" in {
    intercept[IllegalArgumentException] {
      Table(columns = Seq.empty, rows = transactions)
    }.getMessage shouldBe "requirement failed: columns cannot be empty"
  }
  it should "error on empty rows" in {
    intercept[IllegalArgumentException] {
      Table(columns = columns, rows = Seq.empty[Transaction])
    }.getMessage shouldBe "requirement failed: rows cannot be empty"
  }
  it should "error on invalid arity" in {
    intercept[IllegalArgumentException] {
      Table(columns = Seq("a"), rows = transactions)
    }.getMessage shouldBe "requirement failed: columns length must match rows arity (1!=4)"
  }
  it should "pretty print a table" in {
    Table(columns = columns, rows = transactions).prettyString shouldBe
      """|+----------------------------------------+
         || date | amount | source       | status  |
         |+------+--------+--------------+---------+
         || 1    | 4.95   | Cafe Venetia | Success |
         || 2    | 12.65  | Sprout       | Success |
         || 3    | 4.75   | Caltrain     | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "have a pretty toString as well" in {
    val table = Table(columns = columns, rows = transactions)
    table.prettyString shouldBe table.toString
  }
  it should "pretty print a table with a name" in {
    Table(columns = columns, rows = transactions, name = "Transactions").prettyString shouldBe
      """|+----------------------------------------+
         ||              Transactions              |
         |+----------------------------------------+
         || date | amount | source       | status  |
         |+------+--------+--------------+---------+
         || 1    | 4.95   | Cafe Venetia | Success |
         || 2    | 12.65  | Sprout       | Success |
         || 3    | 4.75   | Caltrain     | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "pretty print a table with a name aligned left" in {
    Table(columns = columns, rows = transactions, name = "Transactions", nameAlignment = Left).prettyString shouldBe
      """|+----------------------------------------+
         || Transactions                           |
         |+----------------------------------------+
         || date | amount | source       | status  |
         |+------+--------+--------------+---------+
         || 1    | 4.95   | Cafe Venetia | Success |
         || 2    | 12.65  | Sprout       | Success |
         || 3    | 4.75   | Caltrain     | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "pretty print a table with right column alignment" in {
    Table(columns = columns, rows = transactions, defaultColumnAlignment = Right).prettyString shouldBe
      """|+----------------------------------------+
         || date | amount |       source |  status |
         |+------+--------+--------------+---------+
         ||    1 |   4.95 | Cafe Venetia | Success |
         ||    2 |  12.65 |       Sprout | Success |
         ||    3 |   4.75 |     Caltrain | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "pretty print a table with center column alignment" in {
    Table(columns = columns, rows = transactions, defaultColumnAlignment = Center).prettyString shouldBe
      """|+----------------------------------------+
         || date | amount |    source    | status  |
         |+------+--------+--------------+---------+
         ||  1   |  4.95  | Cafe Venetia | Success |
         ||  2   | 12.65  |    Sprout    | Success |
         ||  3   |  4.75  |   Caltrain   | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "pretty print a table with custom column alignment" in {
    Table(columns = columns, rows = transactions, name = "Transactions",
      nameAlignment = Center, defaultColumnAlignment = Right,
      columnAlignments = Map("date" -> Right, "amount" -> Left, "status" -> Center)
    ).prettyString shouldBe
      """|+----------------------------------------+
         ||              Transactions              |
         |+----------------------------------------+
         || date | amount |       source | status  |
         |+------+--------+--------------+---------+
         ||    1 | 4.95   | Cafe Venetia | Success |
         ||    2 | 12.65  |       Sprout | Success |
         ||    3 | 4.75   |     Caltrain | Pending |
         |+------+--------+--------------+---------+""".stripMargin
  }
  it should "pretty print a table even if data is bad" in {
    val badData1 = Seq(Tuple2(null, "one"), "2" -> "", (null, null), "3" -> Transaction(1, 1.0, "?", "?"))
    Table(columns = Seq("c1", "c2"), rows = badData1, name = "Bad Data").prettyString shouldBe
      """|+-----------------------------+
         ||          Bad Data           |
         |+-----------------------------+
         || c1 | c2                     |
         |+----+------------------------+
         ||    | one                    |
         || 2  |                        |
         ||    |                        |
         || 3  | Transaction(1,1.0,?,?) |
         |+----+------------------------+""".stripMargin
  }
  it should "pretty print a table even if data is really bad" in {
    val badData2 = Seq(null, "", 1).map(Tuple1(_))
    Table(columns = Seq(""), rows = badData2).prettyString shouldBe
      """|+---+
         ||   |
         |+---+
         ||   |
         ||   |
         || 1 |
         |+---+""".stripMargin
  }

}
