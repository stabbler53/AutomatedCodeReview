$(document).ready(function() {
    $("#code-analysis-form").submit(function(e) {
        e.preventDefault();
        $.ajax({
            url: "/code-analysis/",
            type: "POST",
            data: {
                code: $("#code-input").val(),
                csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val()
            },
            success: function(data) {
                $("#analysis-results").html(data.results);
                $("#analysis-summary").html(data.summary);
                $("#analysis-issues").html(data.issues);
            },
            error: function(xhr, textStatus, errorThrown) {
                console.error(textStatus, errorThrown);
            }
        });
    });
});