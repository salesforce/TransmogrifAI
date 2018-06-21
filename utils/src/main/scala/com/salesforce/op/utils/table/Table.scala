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

import com.twitter.algebird.Operators._
import com.twitter.algebird.{Monoid, Semigroup}
import enumeratum._

/**
 * Simple table representation consisting of rows, i.e:
 *
 * +----------------------------------------+
 * |              Transactions              |
 * +----------------------------------------+
 * | date | amount | source       | status  |
 * +------+--------+--------------+---------+
 * | 1    | 4.95   | Cafe Venetia | Success |
 * | 2    | 12.65  | Sprout       | Success |
 * | 3    | 4.75   | Caltrain     | Pending |
 * +------+--------+--------------+---------+
 *
 * @param columns non empty sequence of column names
 * @param rows non empty sequence of rows
 * @param name table name
 * @param nameAlignment table name alignment when printing
 * @param columnAlignments column name & values alignment when printing
 *                         (if not set defaults to [[defaultColumnAlignment]])
 * @param defaultColumnAlignment default column name & values alignment when printing
 * @tparam T row type
 */
case class Table[T <: Product](
  columns: Seq[String],
  rows: Seq[T],
  name: String = "",
  nameAlignment: Alignment = Alignment.Center,
  columnAlignments: Map[String, Alignment] = Map.empty,
  defaultColumnAlignment: Alignment = Alignment.Left
) {
  require(columns.nonEmpty, "columns cannot be empty")
  require(rows.nonEmpty, "rows cannot be empty")
  require(columns.length == rows.head.productArity,
    s"columns length must match rows arity (${columns.length}!=${rows.head.productArity})")

  private implicit val max = Semigroup.from[Int](math.max)
  private implicit val monoid: Monoid[Array[Int]] = Monoid.arrayMonoid[Int]

  private def formatCell(v: String, size: Int, sep: String, fill: String): PartialFunction[Alignment, String] = {
    case Alignment.Left => v + fill * (size - v.length)
    case Alignment.Right => fill * (size - v.length) + v
    case Alignment.Center =>
      String.format("%-" + size + "s", String.format("%" + (v.length + (size - v.length) / 2) + "s", v))
  }

  private def formatRow(
    values: Iterable[String],
    cellSizes: Iterable[Int],
    alignment: String => Alignment = columnAlignments.getOrElse(_, defaultColumnAlignment),
    sep: String = "|",
    fill: String = " "
  ): String = {
    val formatted = values.zipWithIndex.zip(cellSizes).map { case ((v, i), size) =>
      formatCell(v, size, sep, fill)(alignment(columns(i)))
    }
    formatted.mkString(s"$sep$fill", s"$fill$sep$fill", s"$fill$sep")
  }

  /**
   * Pretty print table
   *
   * @return pretty printed table
   */
  def prettyString: String = {
    val rowVals= rows.map(_.productIterator.map(v => Option(v).map(_.toString).getOrElse("")).toSeq)
    val columnSizes = columns.map(c => math.max(1, c.length)).toArray
    val cellSizes = rowVals.map(_.map(_.length).toArray).foldLeft(columnSizes)(_ + _)
    val bracket = formatRow(Seq.fill(cellSizes.length)(""), cellSizes, _ => Alignment.Left, sep = "+", fill = "-")
    val rowWidth = bracket.length - 4
    val cleanBracket = formatRow(Seq(""), Seq(rowWidth), _ => Alignment.Left, sep = "+", fill = "-")
    val maybeName = Option(name) match {
      case Some(n) if n.nonEmpty => Seq(cleanBracket, formatRow(Seq(name), Seq(rowWidth), _ => nameAlignment))
      case _ => Seq.empty
    }
    val columnsHeader = formatRow(columns, cellSizes)
    val formattedRows = rowVals.map(formatRow(_, cellSizes))

    (maybeName ++ Seq(cleanBracket, columnsHeader, bracket) ++ formattedRows :+ bracket).mkString("\n")
  }

  override def toString: String = prettyString

}

sealed trait Alignment extends EnumEntry
object Alignment extends Enum[Alignment] {
  val values = findValues
  case object Left extends Alignment
  case object Right extends Alignment
  case object Center extends Alignment
}

