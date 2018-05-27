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

package com.salesforce.op.cli.gen

import java.io.{File, FileInputStream, FileWriter, InputStream}
import java.nio.file.Files
import java.nio.file.attribute.{PosixFilePermission, PosixFilePermissions}

/**
 * Represents a file that should be created in the new project.
 *
 * @param path The path of the new file to be created, relative to the root of the project
 * @param source The source of the new file, either as a stored string or streaming
 * @param perms The permissions of the file to be created
 */
case class FileInProject(path: String, source: FileSource, perms: FilePermissions = FilePermissions()) {

  /**
   * Create a new ProjectFile by changing this project file's path.
   *
   * @param newExt The new extension
   * @return The changed [[FileInProject]]
   */
  def withExtension(newExt: String): FileInProject = {
    val lastDot = path.lastIndexOf(".")
    val oldBasename = path.substring(0, lastDot)
    val newPath = s"$oldBasename.$newExt"
    FileInProject(newPath, source, perms)
  }

  /**
   * Make this project file be "replaced" from the given file. Replacement means that the given file will be
   * copied into the project directory, with the same name. This is used for copying the avro schema or data csv in the
   * template.
   *
   * @param file The file to copy from
   * @return The changed [[FileInProject]]
   */
  def replaceFrom(file: File): FileInProject = {
    val oldBase = path.substring(0, path.lastIndexOf('/') + 1)
    val finalPath = oldBase + file.getName
    copy(path = finalPath, source = FileSource.Streaming(new FileInputStream(file)))
  }

}

/**
 * Represents the permissions that a new file should have.
 *
 * @param perms The permissions as a Java-compatible set
 */
case class FilePermissions(perms: java.util.Set[PosixFilePermission] = FilePermissions.defaultPerms)

object FilePermissions {
  private val defaultPerms = PosixFilePermissions.fromString("rw-r--r--")
  private val execPerms = PosixFilePermissions.fromString("rwxr-xr-x")

  val default: FilePermissions = FilePermissions(defaultPerms)
  val exec: FilePermissions = FilePermissions(execPerms)
}

/**
 * Represents the "source" of a file that should be created in the new project.
 * This can be an InputStream (usually to stream straight from the CLI jar),
 * or a String (usually from a rendered template).
 */
sealed trait FileSource {
  def writeTo(file: File): Unit
}

object FileSource {

  /**
   * Render the [[FileInProject]] using an in-memory string.
   */
  case class Str(source: String) extends FileSource {
    override def writeTo(file: File): Unit = {
      val writer = new FileWriter(file)
      writer.write(source)
      writer.close()
    }
  }

  /**
   * Render the [[FileInProject]] by copying from an input stream.
   */
  case class Streaming(inputStream: InputStream) extends FileSource {
    override def writeTo(file: File): Unit = {
      Files.copy(inputStream, file.toPath)
    }
  }
}
