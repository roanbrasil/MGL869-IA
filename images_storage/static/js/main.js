function submitLogoutForm() {
  $('#logout-form').submit();
}

$(document).ready(function () {
  $('#select-all').click(function (e) {
    e.preventDefault();
    $('input[name$="delete"]').attr("checked", true);
  })
  $('#unselect-all').click(function (e) {
    e.preventDefault();
    $('input[name$="delete"]').attr("checked", false);
  })
})