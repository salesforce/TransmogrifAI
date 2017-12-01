<%include "header.gsp"%>

    <%include "menu.gsp"%>

    <div class="page-header">
        <h1>Documenation</h1>
    </div>
    <ul>
        <% docs.each {doc ->%>
        <li><a href="/${doc.uri}">${doc.title}</a></li>
        <%}%>
    </ul>


<%include "footer.gsp"%>