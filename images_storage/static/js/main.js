function submitLogoutForm() {
  $('#logout-form').submit();
}

$(document).ready(function () {
  $('#select-all').click(function (e) {
    e.preventDefault();
    $('input[name$="delete"]').prop("checked", true);
  })
  $('#unselect-all').click(function (e) {
    e.preventDefault();
    $('input[name$="delete"]').prop("checked", false);
  })
})